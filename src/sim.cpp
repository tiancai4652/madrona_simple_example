#include "sim.hpp"
#include "sim_debug.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {
// Emit a one-shot init trace. Suppressed when init_log_print_enabled is on,
// or when FCT-only perf mode is enabled, so the [INIT] log dump consumed by
// check/run_parity.py stays free of any extra `[init-trace]` lines that would
// shift line numbers and break diffs. On GPU, only thread 0 prints to avoid
// flooding the 1 MB CUDA printf buffer with 256 duplicates.
static inline void initTrace(const char *msg, bool trace_mode_enabled)
{
    (void)msg;
    (void)trace_mode_enabled;
}

// MWGPU's initECS kernel calls WorldT::registerTypes on device during GPU
// startup, so this implementation must remain available in PTX as well as on
// CPU. Do not stub it out under __CUDA_ARCH__.
void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    registry.registerComponent<SimDriver>();

    registry.registerComponent<PortState>();
    registry.registerComponent<FlowDef>();
    registry.registerComponent<FlowRouteState>();
    registry.registerComponent<FlowRuntimeState>();
    registry.registerComponent<FlowTagState>();
    registry.registerComponent<FlowTagProgress>();
    registry.registerComponent<PortBuffer>();
    registry.registerComponent<DirtyPort>();
    registry.registerComponent<PortCleanup>();
    registry.registerComponent<PortFinishedSourceList>();
    registry.registerComponent<PortOutbox>();
    registry.registerComponent<PortTagLookup>();
    registry.registerComponent<PortDelayedQueue>();
    registry.registerComponent<PortTagList>();
    registry.registerComponent<PortSourceTagList>();
    registry.registerComponent<PortInbox>();
    registry.registerComponent<PortCreateList>();
    registry.registerComponent<PortCompletionList>();
    registry.registerComponent<PortPfcConfig>();
    registry.registerComponent<PortPfcState>();
    registry.registerComponent<PortCachedHints>();
    registry.registerComponent<PortDrainHint>();
    registry.registerComponent<PortTimers>();
    registry.registerComponent<PortTraceLast>();
    registry.registerComponent<IngressTagList>();

    registry.registerSingleton<SimStats>();
    registry.registerSingleton<FlowCounters>();
    registry.registerSingleton<SimRuntimeState>();
    registry.registerSingleton<FlowCompletionBuf>();
    registry.registerSingleton<StepPhaseTimes>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<SimDriverArch>();
    registry.registerArchetype<Port>();
    registry.registerArchetype<FlowMeta>();
    registry.registerArchetype<FlowTag>();

    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
    // GPU-mode introspection mirrors (see types.hpp ExportID comment).
    registry.exportSingleton<SimStats>((uint32_t)ExportID::SimStats);
    registry.exportSingleton<FlowCompletionBuf>(
        (uint32_t)ExportID::FlowCompletionBuf);
    registry.exportSingleton<StepPhaseTimes>(
        (uint32_t)ExportID::StepPhaseTimes);
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      network(init.network),
      maxEpisodeLength(cfg.maxEpisodeLength),
      now(0.0),
      nextPortID(0),
      numTopoNodes(0),
      numTopoLinks(0),
      numPorts(0)
{
    // [init-trace] 用于定位 GPU initWorlds 是否进入、走到哪一步。
    // 噪音抑制 + GPU 单线程打印的细节都封装在 initTrace() 里。
    bool trace_mode_enabled = cfg.perf_fct_only == 0;
    initTrace("Sim::Sim enter", trace_mode_enabled);
    resetNetworkState();
    initTrace("Sim::Sim after resetNetworkState", trace_mode_enabled);
    enableBuffer = cfg.enable_buffer;
    enablePfc = cfg.enable_pfc;
    pfcEgress = cfg.pfc_egress;
    perfFCTOnly = cfg.perf_fct_only;
    defaultLinkDelay = cfg.default_link_delay;
    propagationInterval = cfg.propagation_interval;
    pfcXoffThreshold = cfg.pfc_xoff_threshold;
    pfcXonThreshold = cfg.pfc_xon_threshold;
    dtMin = cfg.dt_min;
    qosMode = cfg.qos_mode;
    for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
        priorWeights[i] = cfg.prior_weights[i];
    }

    Entity agent = ctx.makeEntity<Agent>();
    ctx.get<Reset>(agent) = Reset { .resetNow = 0 };
    ctx.get<Action>(agent) = Action::None;
    ctx.get<GridPos>(agent) = GridPos { .y = 0, .x = 0 };
    ctx.get<Reward>(agent) = Reward { .r = 0.f };
    ctx.get<Done>(agent) = Done { .episodeDone = 0.f };
    ctx.get<CurStep>(agent) = CurStep { .step = 0 };

    Entity driver = ctx.makeEntity<SimDriverArch>();
    ctx.get<SimDriver>(driver) = SimDriver { .tick = 0 };

    initTrace("Sim::Sim before loadTopo", trace_mode_enabled);
    loadTopo(ctx);
    initTrace("Sim::Sim after loadTopo, before loadFlow", trace_mode_enabled);
    loadFlow(ctx);

    // Seed the SimStats / FlowCompletionBuf singleton mirrors so
    // Python-visible getters (num_pending_flows() etc.) work before the
    // first world.step() has run. Without this, GPUImpl reads all-zero
    // mirrors and should_stop(world) returns True immediately, so the
    // main loop skips every step. CPU path reads Sim directly so it
    // never needed this; we still write here so both backends observe
    // identical initial state.
    SimStats &init_stats = ctx.singleton<SimStats>();
    const FlowCounters &flow_counters = ctx.singleton<FlowCounters>();
    const SimRuntimeState &runtime_state = ctx.singleton<SimRuntimeState>();
    init_stats.simulationTime = now;
    init_stats.numFlowDefs = flow_counters.numFlowDefs;
    init_stats.numPendingFlows = flow_counters.numPendingFlows;
    init_stats.numDelayedEvents = runtime_state.numDelayedEvents;
    init_stats.numActiveTags = runtime_state.numActiveTags;
    init_stats.numSourceTags = runtime_state.numSourceTags;
    init_stats.numFlowCompletions = flow_counters.numFlowCompletions;

    FlowCompletionBuf &init_buf = ctx.singleton<FlowCompletionBuf>();
    int32_t n_init = flow_counters.numFlowCompletions;
    if (n_init < 0) n_init = 0;
    if (n_init > MAX_FLOW_COMPLETIONS) n_init = MAX_FLOW_COMPLETIONS;
    int32_t out_idx = 0;
    if (network != nullptr) {
        for (int32_t i = 0; i < network->numFlows && out_idx < n_init; i++) {
            Entity flow_entity = flowMetaEntities[i];
            if (flow_entity == Entity::none()) {
                continue;
            }
            const FlowRuntimeState &runtime =
                ctx.get<FlowRuntimeState>(flow_entity);
            if (runtime.completed == 0) {
                continue;
            }
            init_buf.records[out_idx++] = runtime.completion_record;
        }
    }
    for (int32_t i = out_idx; i < MAX_FLOW_COMPLETIONS; i++) {
        init_buf.records[i] = FlowCompletionRecord {};
    }

    StepPhaseTimes &init_phase_times = ctx.singleton<StepPhaseTimes>();
    init_phase_times = StepPhaseTimes {};

    initTrace("Sim::Sim done", trace_mode_enabled);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
