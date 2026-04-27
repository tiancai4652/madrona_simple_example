#include "sim.hpp"
#include "sim_debug.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {
// Emit a one-shot init trace. Suppressed when init_log_print_enabled is on,
// so the [INIT] log dump consumed by check/run_parity.py stays free of any
// extra `[init-trace]` lines that would shift line numbers and break diffs.
// On GPU, only thread 0 prints to avoid flooding the 1 MB CUDA printf buffer
// with 256 duplicates.
static inline void initTrace(const char *msg)
{
#ifdef MADRONA_GPU_MODE
    if constexpr (init_trace_compiled_in) {
        if (threadIdx.x == 0) {
            printf("[init-trace] %s\n", msg);
        }
    }
#else
    if (!init_log_print_enabled) {
        printf("[init-trace] %s\n", msg);
    }
#endif
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
    registry.registerComponent<FlowTagState>();
    registry.registerComponent<PortBuffer>();

    registry.registerSingleton<SimStats>();
    registry.registerSingleton<FlowCompletionBuf>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<SimDriverArch>();
    registry.registerArchetype<Port>();
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
      numPorts(0),
      numFlowDefs(0),
      numPendingFlows(0),
      numFlowRoutes(0)
{
    // [init-trace] 用于定位 GPU initWorlds 是否进入、走到哪一步。
    // 噪音抑制 + GPU 单线程打印的细节都封装在 initTrace() 里。
    initTrace("Sim::Sim enter");
    resetNetworkState();
    initTrace("Sim::Sim after resetNetworkState");
    enableBuffer = cfg.enable_buffer;
    enablePfc = cfg.enable_pfc;
    pfcEgress = cfg.pfc_egress;
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

    initTrace("Sim::Sim before loadTopo");
    loadTopo(ctx);
    initTrace("Sim::Sim after loadTopo, before loadFlow");
    loadFlow(ctx);

    // Seed the SimStats / FlowCompletionBuf singleton mirrors so
    // Python-visible getters (num_pending_flows() etc.) work before the
    // first world.step() has run. Without this, GPUImpl reads all-zero
    // mirrors and should_stop(world) returns True immediately, so the
    // main loop skips every step. CPU path reads Sim directly so it
    // never needed this; we still write here so both backends observe
    // identical initial state.
    SimStats &init_stats = ctx.singleton<SimStats>();
    init_stats.simulationTime = now;
    init_stats.numFlowDefs = numFlowDefs;
    init_stats.numPendingFlows = numPendingFlows;
    init_stats.numDelayedEvents = numDelayedEvents;
    init_stats.numActiveTags = numTagIndexEntries;
    init_stats.numSourceTags = numSourceTags;
    init_stats.numFlowCompletions = numFlowCompletions;

    FlowCompletionBuf &init_buf = ctx.singleton<FlowCompletionBuf>();
    int32_t n_init = numFlowCompletions;
    if (n_init < 0) n_init = 0;
    if (n_init > MAX_FLOW_COMPLETIONS) n_init = MAX_FLOW_COMPLETIONS;
    for (int32_t i = 0; i < n_init; i++) {
        init_buf.records[i] = flowCompletions[i].record;
    }

    initTrace("Sim::Sim done");
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
