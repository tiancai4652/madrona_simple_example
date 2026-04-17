#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

void scheduleStepSystem(Engine &ctx, SimDriver &driver)
{
    Sim &sim = ctx.data();
    sim.systemLogStep += 1;
    driver.tick += 1;
    sim.schedulePendingFlows();
}

void deliverStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.deliverEvents();
}

void arrivalStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flowArrivalSystem(ctx);
}

void bwUpdateStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.bwUpdateIngressSystem(ctx);
}

void pfcPropagateStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    if (sim.enablePfc != 0) {
        sim.pfcPropagateSystem(ctx);
    }
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
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.allocOnePort(ctx, port_state.port_id, port_state, port_buf, dirty,
        pfc_cfg, pfc_state, hints, drain_hint, cleanup, trace);
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
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.pfcDetectOnePort(ctx, port_state.port_id, port_state, port_buf,
        dirty, pfc_cfg, pfc_state, outbox, trace);
}

// Phase C: per-Port downstream emit. Pushes Arrival/BwUpdate events into
// this port's own PortOutbox; flushPortOutbox later appends them to
// Sim::delayedEvents in port_id ascending order.
void emitOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    DirtyPort &dirty,
    PortOutbox &outbox,
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.emitOnePort(ctx, port_state.port_id, port_state, dirty, outbox, trace);
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

void chooseDTStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.nextDT = sim.chooseDT();
    if (sim.nextDT < 1e-9) {
        sim.nextDT = 0.001;
    }
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
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.advanceOnePortBuffer(ctx, port_state.port_id, sim.nextDT,
        port_state, port_buf, dirty, pfc_state, cleanup, trace);
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

void flowProgressStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flowProgressAndCleanupSystem(ctx, sim.nextDT);
    sim.now += sim.nextDT;
}

}

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

    registry.registerArchetype<Agent>();
    registry.registerArchetype<SimDriverArch>();
    registry.registerArchetype<Port>();
    registry.registerArchetype<FlowTag>();

    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);

    auto n0 = builder.addToGraph<ParallelForNode<Engine,
        scheduleStepSystem, SimDriver>>({});
    auto n1 = builder.addToGraph<ParallelForNode<Engine,
        deliverStepSystem, SimDriver>>({n0});
    auto n2 = builder.addToGraph<ParallelForNode<Engine,
        arrivalStepSystem, SimDriver>>({n1});
    auto n3 = builder.addToGraph<ParallelForNode<Engine,
        bwUpdateStepSystem, SimDriver>>({n2});
    auto n4 = builder.addToGraph<ParallelForNode<Engine,
        pfcPropagateStepSystem, SimDriver>>({n3});
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
        PortCachedHints, PortDrainHint, PortCleanup, PortTraceLast>>({n4});
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
        PortOutbox, PortTraceLast>>({n5});
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
        PortState, DirtyPort, PortOutbox, PortTraceLast>>({n6});
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
        PortPfcState, PortCleanup, PortTraceLast>>({n9});
    auto n10b = builder.addToGraph<ParallelForNode<Engine,
        flushBufferTagCleanupStepSystem, SimDriver>>({n10a});
    auto n10 = builder.addToGraph<ParallelForNode<Engine,
        logBufferTracesStepSystem, SimDriver>>({n10b});
    auto n11 = builder.addToGraph<ParallelForNode<Engine,
        flowProgressStepSystem, SimDriver>>({n10});

#ifdef MADRONA_GPU_MODE
    auto recycle_entities = builder.addToGraph<RecycleEntitiesNode>({n11});
    (void)recycle_entities;
#else
    (void)n11;
#endif
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
    resetNetworkState();
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

    loadTopo(ctx);
    loadFlow(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
