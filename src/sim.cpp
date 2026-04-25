#include "sim.hpp"
#include "sim_debug.hpp"
#include <madrona/mw_gpu_entry.hpp>
#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#endif

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
    if (threadIdx.x == 0) {
        printf("[init-trace] %s\n", msg);
    }
#else
    if (!init_log_print_enabled) {
        printf("[init-trace] %s\n", msg);
    }
#endif
}

// IMPORTANT: must NOT be an anonymous namespace.
//
// All step systems below are passed by-pointer as non-type template arguments
// to ParallelForNode<Engine, &fn, ...>. Per [basic.link], a template
// instantiation that uses an internal-linkage entity becomes itself
// internal-linkage. The Madrona megakernel pipeline (cuda_exec.cpp) parses
// per-TU PTX for `.weak .func _ZN7madrona5mwGPU9userEntry...` symbols and
// stitches them into a single dispatch table. Internal-linkage instantiations
// get mangled into a `_INTERNAL_<hash>_<file>` namespace prefix, so the
// scanner never sees them and our user systems silently disappear from the
// megakernel. Use a NAMED `inline namespace` so the symbols keep external
// linkage (correct mangling) while remaining unqualified-lookup visible in
// `madsimple` for the rest of this TU.
inline namespace systems {

void scheduleStepSystem(Engine &ctx, SimDriver &driver)
{
    Sim &sim = ctx.data();
    sim.systemLogStep += 1;
    driver.tick += 1;
    // [step-trace] SimDriver 只有 1 entity, ParallelForNode 在 megakernel
    // 里只会派 1 个 thread 跑这个 system，但 thread index 不固定，所以
    // 不能用 threadIdx.x == 0 过滤。只用 tick<=3 限频。
#ifdef MADRONA_GPU_MODE
    if (driver.tick <= 3) {
        int32_t t = driver.tick;
        int32_t pend_before = sim.numPendingFlows;
        float now_f = (float)sim.now;
        mwGPU::HostPrint::log(
            "[step-trace] scheduleStepSystem tick=%d pend_before=%d now=%f\n",
            t, pend_before, now_f);
    }
#endif
    sim.schedulePendingFlows();
#ifdef MADRONA_GPU_MODE
    if (driver.tick <= 3) {
        int32_t t = driver.tick;
        int32_t pend_after = sim.numPendingFlows;
        int32_t delayed_after = sim.numDelayedEvents;
        mwGPU::HostPrint::log(
            "[step-trace] scheduleStepSystem AFTER tick=%d pend=%d delayed=%d\n",
            t, pend_after, delayed_after);
    }
#endif
}

void deliverStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.deliverEvents(ctx);
}

// Phase E: per-Port ingress-chain workers. Each operates on a single
// port's inbox + local components only; all cross-port effects are
// deferred to flushTagCreate / flushFlowCompletion / flushPortOutbox /
// flushTagCleanup singletons that follow.
void pfcPropagateOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortPfcState &pfc_state,
    DirtyPort &dirty,
    PortInbox &inbox,
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.pfcPropagateOnePort(ctx, port_state.port_id, port_state, pfc_state,
        dirty, inbox, trace);
}

void arrivalOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    DirtyPort &dirty,
    PortInbox &inbox,
    PortTagList &tag_list,
    PortCreateList &create_list,
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.flowArrivalOnePort(ctx, port_state.port_id, port_state, dirty, inbox,
        tag_list, create_list, trace);
}

void bwUpdateOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortInbox &inbox,
    PortTagList &tag_list,
    PortCreateList &create_list,
    PortCleanup &cleanup,
    PortOutbox &outbox,
    PortCompletionList &completions,
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.bwUpdateOnePort(ctx, port_state.port_id, port_state, port_buf, dirty,
        inbox, tag_list, create_list, cleanup, outbox, completions, trace);
}

void flushTagCreateStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushTagCreate(ctx);
}

void flushFlowCompletionStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushFlowCompletion(ctx);
}

void logIngressChainStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.logIngressChain(ctx);
}

// Phase B.2: per-Port ParallelForNode that runs the bandwidth alloc phase
// for a single port. allocOnePort only touches this port's components plus
// read-only topology/tagIndex/FlowTagState for tags belonging to this port,
// so the fan-out is race-free. Cross-port side-effects (drain timer cache,
// finish-time cache, deferred destroyTag) are collected into the hint /
// cleanup / trace components and flushed by the singletons below.
void allocOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortPfcConfig &pfc_cfg,
    PortPfcState &pfc_state,
    PortCachedHints &hints,
    PortDrainHint &drain_hint,
    PortCleanup &cleanup,
    PortTraceLast &trace,
    PortTagList &tag_list)
{
    Sim &sim = ctx.data();
    sim.allocOnePort(ctx, port_state.port_id, port_state, port_buf, dirty,
        pfc_cfg, pfc_state, hints, drain_hint, cleanup, trace, tag_list);
}

void reduceHintsStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.reducePortCachedHints(ctx);
}

void flushDrainHintsStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortDrainHints(ctx);
}

void flushTagCleanupStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortTagCleanup(ctx);
}

void logAllocTracesStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.logAllocTraces(ctx);
}

// Phase C: per-Port PFC threshold detect. Writes to this port's own
// PortPfcState (want_* deferred timers + pause_active flips),
// PortOutbox (pause/resume events), and PortTraceLast (summary counters).
// Cross-port DirtyPort reads are read-only during this node.
void pfcDetectOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortPfcConfig &pfc_cfg,
    PortPfcState &pfc_state,
    PortOutbox &outbox,
    PortTraceLast &trace,
    PortTagList &tag_list)
{
    Sim &sim = ctx.data();
    sim.pfcDetectOnePort(ctx, port_state.port_id, port_state, port_buf,
        dirty, pfc_cfg, pfc_state, outbox, trace, tag_list);
}

// Phase C: per-Port downstream emit. Pushes Arrival/BwUpdate events into
// this port's own PortOutbox; flushPortOutbox later appends them to
// Sim::delayedEvents in port_id ascending order.
void emitOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    DirtyPort &dirty,
    PortOutbox &outbox,
    PortTraceLast &trace,
    PortTagList &tag_list)
{
    Sim &sim = ctx.data();
    sim.emitOnePort(ctx, port_state.port_id, port_state, dirty, outbox, trace,
        tag_list);
}

// Phase C singletons.
void flushPfcTimersStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortPfcTimers(ctx);
}

void logPfcDetectStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.logPfcDetectTraces(ctx);
}

void flushPortOutboxStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortOutbox(ctx);
}

void logEmitStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.logEmitTraces(ctx);
}

// Phase B.1: per-Port ParallelForNode that snapshots and resets DirtyPort.
// Each port only touches its own DirtyPort / PortTraceLast, so this is safe
// to iterate in parallel on the GPU backend.
void clearDirtyOnePortSystem(Engine &ctx, DirtyPort &dirty, PortTraceLast &trace)
{
    (void)ctx;
    trace.was_dirty_at_clear = (dirty.isDirty != 0) ? 1 : 0;
    dirty.isDirty = 0;
}

// Singleton driven by SimDriver: walks portEntities[] in port_id ascending
// order (the same order the legacy sequential code used) and rebuilds
// lastDirtyPortIDs / emits the clear-summary log line.
void snapshotDirtyStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.snapshotDirtyPorts(ctx);
}

void chooseDTStepSystem(Engine &ctx, SimDriver &driver)
{
    Sim &sim = ctx.data();
    sim.nextDT = sim.chooseDT();
    if (sim.nextDT < 1e-9) {
        sim.nextDT = 0.001;
    }
#ifdef MADRONA_GPU_MODE
    if (driver.tick <= 3) {
        int32_t t = driver.tick;
        float dt_f = (float)sim.nextDT;
        float now_f = (float)sim.now;
        int32_t pend = sim.numPendingFlows;
        int32_t delayed = sim.numDelayedEvents;
        mwGPU::HostPrint::log(
            "[step-trace] chooseDTStepSystem tick=%d nextDT=%f now=%f pend=%d delayed=%d\n",
            t, dt_f, now_f, pend, delayed);
    }
#endif
}

// Phase B.3: per-Port ParallelForNode that advances a single port's
// buffer state for dt = sim.nextDT. Mirrors allocOnePortStepSystem in
// that each port only touches its own components plus read-only
// topology/tagIndex scans; cross-port mutation (destroyTag) is deferred
// into PortCleanup and flushed by the singleton that follows.
void advanceOnePortBufferStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortPfcState &pfc_state,
    PortCleanup &cleanup,
    PortTraceLast &trace,
    PortTagList &tag_list)
{
    Sim &sim = ctx.data();
    sim.advanceOnePortBuffer(ctx, port_state.port_id, sim.nextDT,
        port_state, port_buf, dirty, pfc_state, cleanup, trace, tag_list);
}

void flushBufferTagCleanupStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushBufferTagCleanup(ctx);
}

void logBufferTracesStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.logBufferTraces(ctx);
}

void flowProgressStepSystem(Engine &ctx, SimDriver &driver)
{
    Sim &sim = ctx.data();
#ifdef MADRONA_GPU_MODE
    if (driver.tick <= 3) {
        int32_t t = driver.tick;
        float dt_f = (float)sim.nextDT;
        float now_f = (float)sim.now;
        mwGPU::HostPrint::log(
            "[step-trace] flowProgressStepSystem BEFORE tick=%d nextDT=%f now=%f\n",
            t, dt_f, now_f);
    }
#endif
    sim.flowProgressAndCleanupSystem(ctx, sim.nextDT);
    sim.now += sim.nextDT;
#ifdef MADRONA_GPU_MODE
    if (driver.tick <= 3) {
        int32_t t = driver.tick;
        float now_f = (float)sim.now;
        mwGPU::HostPrint::log(
            "[step-trace] flowProgressStepSystem AFTER tick=%d now=%f\n",
            t, now_f);
    }
#endif
}

// Singleton mirror system. Runs once per step as the very last node so
// the SimStats / FlowCompletionBuf components attached to SimDriverArch
// hold a consistent snapshot of the Sim struct after flowProgress /
// cleanup have settled. mgr.cpp's GPUImpl cudaMemcpy's the exported
// column pointer of these components on every Python-visible getter
// call so the GPU backend has parity with CPU's direct getWorldData()
// access.
void updateSimStatsStepSystem(Engine &ctx,
                              SimDriver &driver,
                              SimStats &stats,
                              FlowCompletionBuf &buf)
{
    Sim &sim = ctx.data();
    stats.simulationTime = sim.now;
    stats.numFlowDefs = sim.numFlowDefs;
    stats.numPendingFlows = sim.numPendingFlows;
    stats.numDelayedEvents = sim.numDelayedEvents;
    stats.numActiveTags = sim.numTagIndexEntries;
    stats.numSourceTags = sim.numSourceTags;
    stats.numFlowCompletions = sim.numFlowCompletions;
    // [step-trace] If host sees lastTick stuck at 0 across many world.step()
    // calls, then this final task graph node never ran. If lastTick climbs
    // but simulationTime stays at 0, the task graph runs but Sim mutations
    // by user systems are not persisted (or sim systems read a different
    // Sim instance than ctx.data() returns).
    stats.lastTick = driver.tick;

    int32_t n = sim.numFlowCompletions;
    if (n < 0) n = 0;
    if (n > MAX_FLOW_COMPLETIONS) n = MAX_FLOW_COMPLETIONS;
    for (int32_t i = 0; i < n; i++) {
        buf.records[i] = sim.flowCompletions[i].record;
    }
}

}  // inline namespace systems

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

    registry.registerComponent<DirtyPort>();
    registry.registerComponent<PortState>();
    registry.registerComponent<FlowTagState>();
    registry.registerComponent<PortBuffer>();
    registry.registerComponent<PortPfcConfig>();
    registry.registerComponent<PortPfcState>();
    registry.registerComponent<PortCachedHints>();
    registry.registerComponent<PortDrainHint>();
    registry.registerComponent<PortCleanup>();
    registry.registerComponent<PortTraceLast>();
    registry.registerComponent<PortOutbox>();
    registry.registerComponent<PortTagList>();
    registry.registerComponent<PortInbox>();
    registry.registerComponent<PortCreateList>();
    registry.registerComponent<PortCompletionList>();

    registry.registerComponent<SimStats>();
    registry.registerComponent<FlowCompletionBuf>();

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
    registry.exportColumn<SimDriverArch, SimStats>(
        (uint32_t)ExportID::SimStats);
    registry.exportColumn<SimDriverArch, FlowCompletionBuf>(
        (uint32_t)ExportID::FlowCompletionBuf);
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    initTrace("Sim::setupTasks enter");
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);

    auto n0 = builder.addToGraph<ParallelForNode<Engine,
        scheduleStepSystem, SimDriver>>({});
    auto n1 = builder.addToGraph<ParallelForNode<Engine,
        deliverStepSystem, SimDriver>>({n0});
    // Phase E: per-Port ingress chain. deliverEvents (n1) already
    // dispatched each due event into the target port's PortInbox and
    // zeroed the per-port PortCreateList / PortCompletionList. We run
    // pfcPropagate → arrival → flushTagCreate → bwUpdate →
    // flushTagCreate → flushTagCleanup → flushFlowCompletion →
    // flushPortOutbox → logIngressChain, matching the effective order
    // the legacy singleton path produced.
    auto n2pfc = builder.addToGraph<ParallelForNode<Engine,
        pfcPropagateOnePortStepSystem,
        PortState, PortPfcState, DirtyPort,
        PortInbox, PortTraceLast>>({n1});
    auto n2arr = builder.addToGraph<ParallelForNode<Engine,
        arrivalOnePortStepSystem,
        PortState, DirtyPort, PortInbox, PortTagList,
        PortCreateList, PortTraceLast>>({n2pfc});
    auto n2createA = builder.addToGraph<ParallelForNode<Engine,
        flushTagCreateStepSystem, SimDriver>>({n2arr});
    auto n2bw = builder.addToGraph<ParallelForNode<Engine,
        bwUpdateOnePortStepSystem,
        PortState, PortBuffer, DirtyPort, PortInbox, PortTagList,
        PortCreateList, PortCleanup, PortOutbox,
        PortCompletionList, PortTraceLast>>({n2createA});
    auto n2createB = builder.addToGraph<ParallelForNode<Engine,
        flushTagCreateStepSystem, SimDriver>>({n2bw});
    auto n2cleanup = builder.addToGraph<ParallelForNode<Engine,
        flushTagCleanupStepSystem, SimDriver>>({n2createB});
    auto n2complete = builder.addToGraph<ParallelForNode<Engine,
        flushFlowCompletionStepSystem, SimDriver>>({n2cleanup});
    auto n2outbox = builder.addToGraph<ParallelForNode<Engine,
        flushPortOutboxStepSystem, SimDriver>>({n2complete});
    auto n4 = builder.addToGraph<ParallelForNode<Engine,
        logIngressChainStepSystem, SimDriver>>({n2outbox});
    // Phase B.2: per-Port fan-out of the bandwidth alloc phase, followed by
    // four SimDriver singletons that fold the hint / cleanup / trace buffers
    // back into global state in port_id ascending order. Sequencing the
    // singletons this way matches the legacy effective-order:
    //   - reducePortCachedHints: cachedNextDrainTime/cachedNextFinishTime
    //   - flushPortDrainHints: backlogDrainTimers clear-then-set
    //   - flushPortTagCleanup: deferred destroyTag in port_id order
    //   - logAllocTraces: alloc-scope log lines
    auto n5a = builder.addToGraph<ParallelForNode<Engine,
        allocOnePortStepSystem,
        PortState, PortBuffer, DirtyPort,
        PortPfcConfig, PortPfcState,
        PortCachedHints, PortDrainHint, PortCleanup, PortTraceLast,
        PortTagList>>({n4});
    auto n5b = builder.addToGraph<ParallelForNode<Engine,
        reduceHintsStepSystem, SimDriver>>({n5a});
    auto n5c = builder.addToGraph<ParallelForNode<Engine,
        flushDrainHintsStepSystem, SimDriver>>({n5b});
    auto n5d = builder.addToGraph<ParallelForNode<Engine,
        flushTagCleanupStepSystem, SimDriver>>({n5c});
    auto n5 = builder.addToGraph<ParallelForNode<Engine,
        logAllocTracesStepSystem, SimDriver>>({n5d});
    // Phase C: per-Port pfcDetect fan-out, followed by singletons that
    // apply deferred want_* PFC timer mutations and emit the
    // pfc_detect summary log in port_id ascending order. We then flush
    // the PFC-phase PortOutbox into Sim::delayedEvents BEFORE the emit
    // per-Port nodes write their own Arrival/BwUpdate events into the
    // same outbox, so the final delayedEvents ordering matches the
    // legacy order (all PFC events in port_id order, then all emit
    // events in port_id order).
    auto n6a = builder.addToGraph<ParallelForNode<Engine,
        pfcDetectOnePortStepSystem,
        PortState, PortBuffer, DirtyPort,
        PortPfcConfig, PortPfcState,
        PortOutbox, PortTraceLast,
        PortTagList>>({n5});
    auto n6b = builder.addToGraph<ParallelForNode<Engine,
        flushPfcTimersStepSystem, SimDriver>>({n6a});
    auto n6c = builder.addToGraph<ParallelForNode<Engine,
        flushPortOutboxStepSystem, SimDriver>>({n6b});
    auto n6 = builder.addToGraph<ParallelForNode<Engine,
        logPfcDetectStepSystem, SimDriver>>({n6c});
    // Phase C: per-Port downstream emit fan-out, followed by an outbox
    // flush and the emit summary log singleton.
    auto n7a = builder.addToGraph<ParallelForNode<Engine,
        emitOnePortStepSystem,
        PortState, DirtyPort, PortOutbox, PortTraceLast,
        PortTagList>>({n6});
    auto n7b = builder.addToGraph<ParallelForNode<Engine,
        flushPortOutboxStepSystem, SimDriver>>({n7a});
    auto n7 = builder.addToGraph<ParallelForNode<Engine,
        logEmitStepSystem, SimDriver>>({n7b});
    // Phase B.1: per-Port clearDirtyOnePortSystem fans out over every Port
    // entity; the follow-up SimDriver singleton snapshotDirtyStepSystem
    // collapses the per-port snapshots into lastDirtyPortIDs in
    // port_id ascending order and emits the clear-summary log.
    auto n8a = builder.addToGraph<ParallelForNode<Engine,
        clearDirtyOnePortSystem, DirtyPort, PortTraceLast>>({n7});
    auto n8 = builder.addToGraph<ParallelForNode<Engine,
        snapshotDirtyStepSystem, SimDriver>>({n8a});
    auto n9 = builder.addToGraph<ParallelForNode<Engine,
        chooseDTStepSystem, SimDriver>>({n8});
    // Phase B.3: per-Port fan-out of the buffer advance phase, followed by
    // two SimDriver singletons. flushBufferTagCleanup replays deferred
    // destroyTag in port_id ascending order; logBufferTraces sums the
    // per-port buffer summary fields before we advance `now`.
    auto n10a = builder.addToGraph<ParallelForNode<Engine,
        advanceOnePortBufferStepSystem,
        PortState, PortBuffer, DirtyPort,
        PortPfcState, PortCleanup, PortTraceLast,
        PortTagList>>({n9});
    auto n10b = builder.addToGraph<ParallelForNode<Engine,
        flushBufferTagCleanupStepSystem, SimDriver>>({n10a});
    auto n10 = builder.addToGraph<ParallelForNode<Engine,
        logBufferTracesStepSystem, SimDriver>>({n10b});
    auto n11 = builder.addToGraph<ParallelForNode<Engine,
        flowProgressStepSystem, SimDriver>>({n10});

    // Final mirror pass: snapshot Sim scalar counts + flowCompletions
    // into the SimStats / FlowCompletionBuf singleton components on
    // SimDriverArch. mgr.cpp's GPUImpl reads these via exported columns.
    // Runs unconditionally on both backends; CPU does not need it but
    // the cost is one archetype iteration over a single entity.
    auto n12 = builder.addToGraph<ParallelForNode<Engine,
        updateSimStatsStepSystem,
        SimDriver, SimStats, FlowCompletionBuf>>({n11});

#ifdef MADRONA_GPU_MODE
    auto recycle_entities = builder.addToGraph<RecycleEntitiesNode>({n12});
    (void)recycle_entities;
#else
    (void)n12;
#endif
    initTrace("Sim::setupTasks done");
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

    // Seed the SimStats / FlowCompletionBuf mirror on SimDriverArch so
    // Python-visible getters (num_pending_flows() etc.) work before the
    // first world.step() has run. Without this, GPUImpl reads all-zero
    // mirrors and should_stop(world) returns True immediately, so the
    // main loop skips every step. CPU path reads Sim directly so it
    // never needed this; we still write here so both backends observe
    // identical initial state.
    SimStats &init_stats = ctx.get<SimStats>(driver);
    init_stats.simulationTime = now;
    init_stats.numFlowDefs = numFlowDefs;
    init_stats.numPendingFlows = numPendingFlows;
    init_stats.numDelayedEvents = numDelayedEvents;
    init_stats.numActiveTags = numTagIndexEntries;
    init_stats.numSourceTags = numSourceTags;
    init_stats.numFlowCompletions = numFlowCompletions;

    FlowCompletionBuf &init_buf = ctx.get<FlowCompletionBuf>(driver);
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
