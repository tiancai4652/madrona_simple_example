#include "sim.hpp"
#include "sim_debug.hpp"

#ifndef MADRONA_GPU_MODE
#include <chrono>
#endif

using namespace madrona;
using namespace madrona::math;

namespace madsimple {
// Retained as a no-op so existing call sites stay minimal.
static inline void initTrace(const char *msg, bool trace_mode_enabled)
{
    (void)msg;
    (void)trace_mode_enabled;
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

#ifndef MADRONA_GPU_MODE
inline double hostWallNowSeconds()
{
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(
        now.time_since_epoch()).count();
}

inline void beginHostStepPhase(Engine &ctx, SimDriver &driver)
{
    StepPhaseTimes &times = ctx.singleton<StepPhaseTimes>();
    times = StepPhaseTimes {};
    times.step = driver.tick + 1;
    double now_s = hostWallNowSeconds();
    times._stepStartWallTimeS = now_s;
    times._lastBoundaryWallTimeS = now_s;
}

inline void closeHostStepPhase(Engine &ctx, StepPhaseID phase)
{
    StepPhaseTimes &times = ctx.singleton<StepPhaseTimes>();
    double now_s = hostWallNowSeconds();
    double delta_s = now_s - times._lastBoundaryWallTimeS;
    if (delta_s < 0.0) {
        delta_s = 0.0;
    }
    times.phaseWallTimeS[(uint32_t)phase] += delta_s;
    times._lastBoundaryWallTimeS = now_s;
}

inline void finalizeHostStepPhase(Engine &ctx)
{
    StepPhaseTimes &times = ctx.singleton<StepPhaseTimes>();
    double now_s = hostWallNowSeconds();
    double delta_s = now_s - times._lastBoundaryWallTimeS;
    if (delta_s < 0.0) {
        delta_s = 0.0;
    }
    times.phaseWallTimeS[(uint32_t)StepPhaseID::BufferProgress] += delta_s;
    times._lastBoundaryWallTimeS = now_s;
    times.totalWallTimeS = now_s - times._stepStartWallTimeS;
}
#else
inline void beginHostStepPhase(Engine &, SimDriver &) {}
inline void closeHostStepPhase(Engine &, StepPhaseID) {}
inline void finalizeHostStepPhase(Engine &) {}
#endif

// Compile-time diagnostic switch for bisecting final NVVM link pressure.
// Stage meanings in setupTasks():
//   4 = through ingress chain
//   5 = through alloc
//   6 = through pfc
//   7 = through emit
//   9 = through clear/chooseDT
//  12 = full graph
constexpr int kTaskgraphStageLimit = 12;

MADRONA_NO_INLINE void beginScheduleStepSystem(Engine &ctx, SimDriver &driver)
{
    beginHostStepPhase(ctx, driver);

    Sim &sim = ctx.data();
    if (sim.traceModeEnabled()) {
        sim.systemLogStep += 1;
    }
    SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    runtime.delayedDropCount = 0;
    runtime.delayedPfcDropCount = 0;
    driver.tick += 1;
}

MADRONA_NO_INLINE void preparePendingFlowMetaStepSystem(
    Engine &ctx,
    FlowDef &flow,
    FlowRuntimeState &runtime,
    FlowScheduleState &schedule_state)
{
    Sim &sim = ctx.data();
    sim.preparePendingFlowMeta(ctx, flow, runtime, schedule_state);
}

MADRONA_NO_INLINE void flushScheduleStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.schedulePendingFlows(ctx);
}

MADRONA_NO_INLINE void beginDeliverStepSystem(Engine &ctx, SimDriver &)
{
    closeHostStepPhase(ctx, StepPhaseID::Schedule);
    Sim &sim = ctx.data();
    if (sim.traceModeEnabled() &&
        compiledSystemLogEnabled("ingress_chain", sim.systemLogStep)) {
        printSystemBegin(sim.systemLogStep, sim.now,
            "ingress_chain", "deliver_events");
    }
}

MADRONA_NO_INLINE void deliverOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortDelayedQueue &queue,
    PortInbox &inbox,
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.deliverEventsOnePort(ctx, port_state.port_id, queue, inbox, trace);
}

MADRONA_NO_INLINE void postDeliverStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.finishDeliverEvents(ctx);
}

MADRONA_NO_INLINE void resetIngressPortStateStepSystem(
    Engine &ctx,
    PortInbox &inbox,
    PortCreateList &create_list,
    PortCompletionList &completion_list)
{
    (void)ctx;
    inbox.num_arrival = 0;
    inbox.num_bwupd = 0;
    inbox.num_pfc = 0;
    create_list.num = 0;
    completion_list.num = 0;
}

// Phase E: per-Port ingress-chain workers. Each operates on a single
// port's inbox + local components only; cross-port effects are reduced
// to small deferred fold-back passes (ingress-link replay,
// flushFlowCompletion / flushPortOutbox / flushTagCleanup) that follow.
MADRONA_NO_INLINE void pfcPropagateOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    DirtyPort &dirty,
    PortPfcState &pfc_state,
    PortTraceLast &trace,
    PortInbox &inbox)
{
    Sim &sim = ctx.data();
    sim.pfcPropagateOnePort(ctx, port_state.port_id, port_state, pfc_state,
        dirty, inbox, trace);
}

MADRONA_NO_INLINE void arrivalOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    DirtyPort &dirty,
    PortTraceLast &trace,
    PortInbox &inbox,
    PortTagList &tag_list,
    PortSourceTagList &source_tag_list,
    PortCreateList &create_list)
{
    Sim &sim = ctx.data();
    sim.flowArrivalOnePort(ctx, port_state.port_id, port_state, dirty, inbox,
        tag_list, source_tag_list, create_list, trace);
}

MADRONA_NO_INLINE void bwUpdateOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortTraceLast &trace,
    PortInbox &inbox,
    PortTagList &tag_list,
    PortCreateList &create_list,
    PortCleanup &cleanup,
    PortOutbox &outbox,
    PortCompletionList &completions)
{
    Sim &sim = ctx.data();
    sim.bwUpdateOnePort(ctx, port_state.port_id, port_state, port_buf, dirty,
        inbox, tag_list, create_list, cleanup, outbox, completions, trace);
}

MADRONA_NO_INLINE void materializeTagCreateOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    DirtyPort &dirty,
    PortCreateList &create_list,
    PortIngressLinkList &ingress_links,
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.materializeTagCreateOnePort(ctx, port_state.port_id, dirty,
        create_list, ingress_links, trace);
}

MADRONA_NO_INLINE void flushIngressTagLinksStepSystem(
    Engine &ctx,
    SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushIngressTagLinks(ctx);
}

MADRONA_NO_INLINE void flushIngressTagUnlinksStepSystem(
    Engine &ctx,
    SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushIngressTagUnlinks(ctx);
}

MADRONA_NO_INLINE void materializeTagCleanupOnePortNowStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortCleanup &cleanup,
    PortIngressUnlinkList &ingress_unlinks,
    PortCompletionList &completions,
    PortOutbox &outbox)
{
    Sim &sim = ctx.data();
    sim.materializeTagCleanupOnePort(ctx, port_state.port_id, cleanup,
        ingress_unlinks, completions, outbox, sim.now);
}

MADRONA_NO_INLINE void materializeTagCleanupOnePortFrameEndStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortCleanup &cleanup,
    PortIngressUnlinkList &ingress_unlinks,
    PortCompletionList &completions,
    PortOutbox &outbox)
{
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.materializeTagCleanupOnePort(ctx, port_state.port_id, cleanup,
        ingress_unlinks, completions, outbox, sim.now + runtime.nextDT);
}

MADRONA_NO_INLINE void postIngressStepSystem(Engine &ctx, SimDriver &)
{
    closeHostStepPhase(ctx, StepPhaseID::Deliver);
    Sim &sim = ctx.data();
    sim.flushIngressTagUnlinks(ctx);
    sim.flushFlowCompletion(ctx);
    sim.flushPortOutbox(ctx);
    if (sim.traceModeEnabled()) {
        sim.logIngressChain(ctx);
    }
}

// Phase B.2: per-Port ParallelForNode that runs the bandwidth alloc phase
// for a single port. allocOnePort only touches this port's components plus
// read-only topology/tagIndex/FlowTagState for tags belonging to this port,
// so the fan-out is race-free. Cross-port side-effects (drain timer cache,
// finish-time cache, deferred destroyTag) are collected into the hint /
// cleanup / trace components and flushed by the singletons below.
MADRONA_NO_INLINE void allocOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortPfcConfig &pfc_cfg,
    PortPfcState &pfc_state,
    PortCachedHints &hints,
    PortDrainHint &drain_hint,
    PortTraceLast &trace,
    PortCleanup &cleanup,
    PortTagList &tag_list)
{
    Sim &sim = ctx.data();
    sim.allocOnePort(ctx, port_state.port_id, port_state, port_buf, dirty,
        pfc_cfg, pfc_state, hints, drain_hint, cleanup, trace, tag_list);
}

MADRONA_NO_INLINE void flushTagCleanupStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortTagCleanup(ctx);
}

MADRONA_NO_INLINE void reducePortCachedHintsStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.reducePortCachedHints(ctx);
}

MADRONA_NO_INLINE void applyDrainHintOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortDrainHint &hint,
    PortTimers &timers)
{
    Sim &sim = ctx.data();
    sim.applyDrainHintOnePort(port_state.port_id, hint, timers);
}

MADRONA_NO_INLINE void postAllocStepSystem(Engine &ctx, SimDriver &)
{
    closeHostStepPhase(ctx, StepPhaseID::Ingress);
    Sim &sim = ctx.data();
    sim.flushIngressTagUnlinks(ctx);
    sim.flushFlowCompletion(ctx);
    sim.flushPortOutbox(ctx);
    if (sim.traceModeEnabled()) {
        sim.logAllocTraces(ctx);
    }
}

// Phase C: per-Port PFC threshold detect. Writes to this port's own
// PortPfcState (want_* deferred timers + pause_active flips),
// PortOutbox (pause/resume events), and PortTraceLast (summary counters).
// Cross-port DirtyPort reads are read-only during this node.
MADRONA_NO_INLINE void pfcDetectOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortPfcConfig &pfc_cfg,
    PortPfcState &pfc_state,
    PortTraceLast &trace,
    PortOutbox &outbox,
    PortTagList &tag_list,
    IngressTagList &ingress_list)
{
    Sim &sim = ctx.data();
    sim.pfcDetectOnePort(ctx, port_state.port_id, port_state, port_buf,
        dirty, pfc_cfg, pfc_state, outbox, trace, tag_list, ingress_list);
}

MADRONA_NO_INLINE void markPfcIngressCheckTargetsStepSystem(
    Engine &ctx,
    SimDriver &)
{
    Sim &sim = ctx.data();
    sim.markPfcIngressCheckTargets(ctx);
}

// Phase C: per-Port downstream emit. Pushes Arrival/BwUpdate events into
// this port's own PortOutbox; flushPortOutbox later appends them to
// Sim::delayedEvents in port_id ascending order.
MADRONA_NO_INLINE void emitOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    DirtyPort &dirty,
    PortTraceLast &trace,
    PortOutbox &outbox,
    PortTagList &tag_list)
{
    Sim &sim = ctx.data();
    sim.emitOnePort(ctx, port_state.port_id, port_state, dirty, outbox, trace,
        tag_list);
}

// Phase C singletons.
MADRONA_NO_INLINE void applyPfcTimerOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortPfcState &state,
    PortTimers &timers)
{
    Sim &sim = ctx.data();
    sim.applyPfcTimerOnePort(port_state.port_id, state, timers);
}

MADRONA_NO_INLINE void postPfcStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortOutbox(ctx);
    if (sim.traceModeEnabled()) {
        sim.logPfcDetectTraces(ctx);
        sim.logPfcDebugTraces(ctx);
    }
}

MADRONA_NO_INLINE void postEmitStepSystem(Engine &ctx, SimDriver &)
{
    closeHostStepPhase(ctx, StepPhaseID::Alloc);
    Sim &sim = ctx.data();
    sim.flushPortOutbox(ctx);
    if (sim.traceModeEnabled()) {
        sim.logEmitTraces(ctx);
    }
}

// Phase B.1: per-Port ParallelForNode that snapshots and resets DirtyPort.
// Each port only touches its own DirtyPort / PortTraceLast, so this is safe
// to iterate in parallel on the GPU backend.
MADRONA_NO_INLINE void clearDirtyOnePortSystem(Engine &ctx,
                                               DirtyPort &dirty,
                                               PortTraceLast &trace)
{
    (void)ctx;
    trace.was_dirty_at_clear = (dirty.isDirty != 0) ? 1 : 0;
    dirty.isDirty = 0;
}

MADRONA_NO_INLINE void snapshotDirtyOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortDelayedQueue &queue,
    PortTimers &timers,
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    sim.snapshotDirtyOnePort(port_state.port_id, queue, timers, trace);
}

// Singleton driven by SimDriver: counts the per-port was_dirty_at_clear
// snapshots for logging, then computes nextDT for the upcoming buffer step.
// The buffer phase now consumes was_dirty_at_clear directly on each Port, so
// no Sim-global dirty-port list is rebuilt here.
MADRONA_NO_INLINE void postClearStepSystem(Engine &ctx, SimDriver &driver)
{
    (void)driver;
    closeHostStepPhase(ctx, StepPhaseID::PfcEmit);
    Sim &sim = ctx.data();
    SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.snapshotDirtyPorts(ctx);
    runtime.nextDT = sim.chooseDT(ctx);
    if (runtime.nextDT < 1e-9) {
        runtime.nextDT = 0.001;
    }
}

// Phase B.3: per-Port ParallelForNode that advances a single port's
// buffer state for dt = sim.nextDT. Mirrors allocOnePortStepSystem in
// that each port only touches its own components plus read-only
// topology/tagIndex scans; cross-port mutation (destroyTag) is deferred
// into PortCleanup and flushed by the singleton that follows.
MADRONA_NO_INLINE void advanceOnePortBufferStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortPfcState &pfc_state,
    PortTimers &timers,
    PortTraceLast &trace,
    PortCleanup &cleanup,
    PortTagList &tag_list)
{
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.advanceOnePortBuffer(ctx, port_state.port_id, runtime.nextDT,
        port_state, port_buf, dirty, pfc_state, timers, cleanup, trace,
        tag_list);
}

MADRONA_NO_INLINE void progressFinishedSourcesOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortTagList &tag_list,
    PortSourceTagList &source_tag_list,
    PortCachedHints &hints,
    PortFinishedSourceList &finished_list,
    PortTraceLast &trace,
    PortOutbox &outbox)
{
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.progressFinishedSourcesOnePort(ctx, runtime.nextDT,
        sim.now + runtime.nextDT,
        port_state.port_id, port_state, tag_list, source_tag_list, hints,
        finished_list, trace, outbox);
}

MADRONA_NO_INLINE void cleanupFinishedSourcesOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortCachedHints &hints,
    PortFinishedSourceList &finished_list)
{
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.cleanupFinishedSourcesOnePort(ctx, sim.now + runtime.nextDT,
        port_state.port_id, hints, finished_list);
}

MADRONA_NO_INLINE void progressBacklogDrainTimerOnePortStepSystem(
    Engine &ctx,
    DirtyPort &dirty,
    PortTimers &timers)
{
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.progressBacklogDrainTimerOnePort(timers, dirty, runtime.nextDT);
}

MADRONA_NO_INLINE void flushProgressOutboxStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortOutbox(ctx);
}

MADRONA_NO_INLINE void progressPfcTimerOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortTimers &timers,
    IngressTagList &ingress_list,
    PortDirtyMarkList &dirty_marks)
{
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.progressPfcTimerOnePort(ctx, port_state.port_id, timers,
        ingress_list, dirty_marks, runtime.nextDT);
}

MADRONA_NO_INLINE void flushDirtyPortMarksStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushDirtyPortMarks(ctx);
}

MADRONA_NO_INLINE void prepareProgressStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.prepareProgressState(ctx);
}

MADRONA_NO_INLINE void markBufferedPortDirtyOnePortStepSystem(
    Engine &ctx,
    PortBuffer &port_buf,
    DirtyPort &dirty,
    PortTraceLast &trace)
{
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    trace.progress_buffered_dirty_marked = 0;
    if (runtime.progressAllExhausted != 0) {
        sim.markBufferedPortDirtyOnePort(port_buf, dirty, trace);
    }
}

MADRONA_NO_INLINE void finishProgressStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.finishProgressState(ctx, runtime.nextDT);
}

MADRONA_NO_INLINE void postBufferStepSystem(Engine &ctx, SimDriver &driver)
{
    closeHostStepPhase(ctx, StepPhaseID::ClearDT);
    Sim &sim = ctx.data();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    sim.flushIngressTagUnlinks(ctx);
    sim.flushFlowCompletion(ctx);
    sim.flushPortOutbox(ctx);
    if (sim.traceModeEnabled()) {
        sim.logBufferTraces(ctx);
    }
    sim.refreshTagCounters(ctx);
    sim.now += runtime.nextDT;
    SimStats &stats = ctx.singleton<SimStats>();
    const FlowCounters &flow_counters = ctx.singleton<FlowCounters>();
    FlowCompletionBuf &buf = ctx.singleton<FlowCompletionBuf>();
    stats.simulationTime = sim.now;
    stats.numFlowDefs = flow_counters.numFlowDefs;
    stats.numPendingFlows = flow_counters.numPendingFlows;
    stats.numDelayedEvents = runtime.numDelayedEvents;
    stats.numActiveTags = runtime.numActiveTags;
    stats.numSourceTags = runtime.numSourceTags;
    stats.numFlowCompletions = flow_counters.numFlowCompletions;
    stats.lastTick = driver.tick;

    finalizeHostStepPhase(ctx);

    int32_t n = flow_counters.numFlowCompletions;
    if (n < 0) n = 0;
    if (n > MAX_FLOW_COMPLETIONS) n = MAX_FLOW_COMPLETIONS;
    int32_t out_idx = 0;
    if (sim.network != nullptr) {
        for (int32_t i = 0; i < sim.network->numFlows && out_idx < n; i++) {
            Entity flow_entity = sim.flowMetaEntities[i];
            if (flow_entity == Entity::none()) {
                continue;
            }
            const FlowRuntimeState &flow_runtime =
                ctx.get<FlowRuntimeState>(flow_entity);
            if (flow_runtime.completed == 0) {
                continue;
            }
            FlowCompletionRecord record = flow_runtime.completion_record;
            const FlowDef &flow_def = ctx.get<FlowDef>(flow_entity);
            bool cross_leaf = flow_def.src_node / 32 != flow_def.dst_node / 32;
            if (runtime.numActiveTags == 0 && cross_leaf &&
                record.end_time < sim.now) {
                record.end_time = sim.now;
            }
            buf.records[out_idx++] = record;
        }
    }
    for (int32_t i = out_idx; i < MAX_FLOW_COMPLETIONS; i++) {
        buf.records[i] = FlowCompletionRecord {};
    }
}

}  // inline namespace systems

// MWGPU's initTasks kernel executes WorldT::setupTasks on device to build the
// task graph and instantiate the userEntry / UserFuncID symbols scanned by
// cuda_exec.cpp. Stubbing this out under __CUDA_ARCH__ silently removes every
// user system from the megakernel and leaves the GPU with no task graph.
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &cfg)
{
    bool trace_mode_enabled = cfg.perf_fct_only == 0;
    initTrace("Sim::setupTasks enter", trace_mode_enabled);
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);

    auto n0begin = builder.addToGraph<ParallelForNode<Engine,
        beginScheduleStepSystem, SimDriver>>({});
    auto n0prepare = builder.addToGraph<ParallelForNode<Engine,
        preparePendingFlowMetaStepSystem,
        FlowDef, FlowRuntimeState, FlowScheduleState>>({n0begin});
    auto n0 = builder.addToGraph<ParallelForNode<Engine,
        flushScheduleStepSystem, SimDriver>>({n0prepare});
    auto n1reset = builder.addToGraph<ParallelForNode<Engine,
        resetIngressPortStateStepSystem,
        PortInbox, PortCreateList, PortCompletionList>>({n0});
    auto n1begin = builder.addToGraph<ParallelForNode<Engine,
        beginDeliverStepSystem, SimDriver>>({n1reset});
    auto n1deliver = builder.addToGraph<ParallelForNode<Engine,
        deliverOnePortStepSystem,
        PortState, PortDelayedQueue, PortInbox, PortTraceLast>>({n1begin});
    auto n1 = builder.addToGraph<ParallelForNode<Engine,
        postDeliverStepSystem, SimDriver>>({n1deliver});
    // Phase E: per-Port ingress chain. resetIngressPortStateStepSystem
    // has already cleared each port's inbox / create / completion
    // scratch, and deliverOnePort (n1deliver) has dispatched each due
    // event from the port-local future queues into that port's PortInbox.
    // We run arrival → per-port tag materialize → flushIngressTagLinks →
    // bwUpdate → per-port tag materialize → flushIngressTagLinks →
    // pfcPropagate →
    // flushIngressTagLinks → flushTagCleanup → flushFlowCompletion →
    // flushPortOutbox → logIngressChain, matching the legacy semantic
    // order while removing the cross-port write from createTagOnPort.
    auto n2arr = builder.addToGraph<ParallelForNode<Engine,
        arrivalOnePortStepSystem,
        PortState, DirtyPort, PortTraceLast, PortInbox, PortTagList,
        PortSourceTagList, PortCreateList>>({n1});
    auto n2createA_local = builder.addToGraph<ParallelForNode<Engine,
        materializeTagCreateOnePortStepSystem,
        PortState, DirtyPort, PortCreateList, PortIngressLinkList,
        PortTraceLast>>({n2arr});
    auto n2createA = builder.addToGraph<ParallelForNode<Engine,
        flushIngressTagLinksStepSystem, SimDriver>>({n2createA_local});
    auto n2bw = builder.addToGraph<ParallelForNode<Engine,
        bwUpdateOnePortStepSystem,
        PortState, PortBuffer, DirtyPort, PortTraceLast, PortInbox,
        PortTagList, PortCreateList, PortCleanup, PortOutbox,
        PortCompletionList>>({n2createA});
    auto n2createB_local = builder.addToGraph<ParallelForNode<Engine,
        materializeTagCreateOnePortStepSystem,
        PortState, DirtyPort, PortCreateList, PortIngressLinkList,
        PortTraceLast>>({n2bw});
    auto n2createB = builder.addToGraph<ParallelForNode<Engine,
        flushIngressTagLinksStepSystem, SimDriver>>({n2createB_local});
    auto n2pfc = builder.addToGraph<ParallelForNode<Engine,
        pfcPropagateOnePortStepSystem,
        PortState, DirtyPort, PortPfcState, PortTraceLast, PortInbox>>(
            {n2createB});
    auto n2cleanup_local = builder.addToGraph<ParallelForNode<Engine,
        materializeTagCleanupOnePortNowStepSystem,
        PortState, PortCleanup, PortIngressUnlinkList,
        PortCompletionList, PortOutbox>>({n2pfc});
    auto n2cleanup = builder.addToGraph<ParallelForNode<Engine,
        flushIngressTagUnlinksStepSystem, SimDriver>>({n2cleanup_local});
    auto n4 = builder.addToGraph<ParallelForNode<Engine,
        postIngressStepSystem, SimDriver>>({n2cleanup});
    TaskGraphNodeID final_node = n4;
    // Phase B.2: per-Port fan-out of the bandwidth alloc phase, followed by
    // a singleton reduction for cachedNextDrainTime / cachedNextFinishTime,
    // then a per-Port drain-hint apply pass, then the remaining singleton
    // cleanup / logging fold-back. backlogDrainTimers is now indexed directly
    // by port_id, so applying each port's PortDrainHint is race-free.
    if constexpr (kTaskgraphStageLimit >= 5) {
    auto n5a = builder.addToGraph<ParallelForNode<Engine,
        allocOnePortStepSystem,
        PortState, PortBuffer, DirtyPort, PortPfcConfig, PortPfcState,
        PortCachedHints, PortDrainHint, PortTraceLast, PortCleanup,
        PortTagList>>({n4});
    auto n5reduce = builder.addToGraph<ParallelForNode<Engine,
        reducePortCachedHintsStepSystem, SimDriver>>({n5a});
    auto n5drain = builder.addToGraph<ParallelForNode<Engine,
        applyDrainHintOnePortStepSystem,
        PortState, PortDrainHint, PortTimers>>({n5reduce});
    auto n5cleanup = builder.addToGraph<ParallelForNode<Engine,
        materializeTagCleanupOnePortNowStepSystem,
        PortState, PortCleanup, PortIngressUnlinkList,
        PortCompletionList, PortOutbox>>({n5drain});
    auto n5 = builder.addToGraph<ParallelForNode<Engine,
        postAllocStepSystem, SimDriver>>({n5cleanup});
    final_node = n5;
    // Phase C: per-Port pfcDetect fan-out, followed by a per-Port timer
    // apply pass, then the singleton outbox flush / summary log. PFC
    // pause/resume timers are now port-indexed, so applying each port's
    // deferred want_* state is race-free. We still flush the PFC-phase
    // PortOutbox into Sim::delayedEvents BEFORE the emit per-Port nodes
    // write their own Arrival/BwUpdate events into the same outbox, so
    // delayedEvents ordering remains legacy-compatible.
    if constexpr (kTaskgraphStageLimit >= 6) {
    auto n6mark = builder.addToGraph<ParallelForNode<Engine,
        markPfcIngressCheckTargetsStepSystem, SimDriver>>({n5});
    auto n6a = builder.addToGraph<ParallelForNode<Engine,
        pfcDetectOnePortStepSystem,
        PortState, PortBuffer, DirtyPort, PortPfcConfig, PortPfcState,
        PortTraceLast, PortOutbox, PortTagList, IngressTagList>>({n6mark});
    auto n6timers = builder.addToGraph<ParallelForNode<Engine,
        applyPfcTimerOnePortStepSystem,
        PortState, PortPfcState, PortTimers>>({n6a});
    auto n6 = builder.addToGraph<ParallelForNode<Engine,
        postPfcStepSystem, SimDriver>>({n6timers});
    final_node = n6;
    // Phase C: per-Port downstream emit fan-out, followed by an outbox
    // flush and the emit summary log singleton.
    if constexpr (kTaskgraphStageLimit >= 7) {
    auto n7a = builder.addToGraph<ParallelForNode<Engine,
        emitOnePortStepSystem,
        PortState, DirtyPort, PortTraceLast, PortOutbox, PortTagList>>({n6});
    auto n7 = builder.addToGraph<ParallelForNode<Engine,
        postEmitStepSystem, SimDriver>>({n7a});
    final_node = n7;
    // Phase B.1: per-Port clearDirtyOnePortSystem fans out over every Port
    // entity; the follow-up SimDriver singleton emits the clear-summary log
    // and computes nextDT. The per-port was_dirty_at_clear flags stay on the
    // Port entities and are consumed directly by the buffer phase.
    if constexpr (kTaskgraphStageLimit >= 9) {
    auto n8a = builder.addToGraph<ParallelForNode<Engine,
        clearDirtyOnePortSystem, DirtyPort, PortTraceLast>>({n7});
    auto n8snapshot = builder.addToGraph<ParallelForNode<Engine,
        snapshotDirtyOnePortStepSystem,
        PortState, PortDelayedQueue, PortTimers, PortTraceLast>>({n8a});
    auto n9 = builder.addToGraph<ParallelForNode<Engine,
        postClearStepSystem, SimDriver>>({n8snapshot});
    final_node = n9;
    // Phase B.3: per-Port fan-out of the buffer advance phase, followed by
    // two SimDriver singletons. flushBufferTagCleanup replays deferred
    // destroyTag in port_id ascending order; logBufferTraces sums the
    // per-port buffer summary fields before we advance `now`.
    if constexpr (kTaskgraphStageLimit >= 12) {
    auto n10a = builder.addToGraph<ParallelForNode<Engine,
        advanceOnePortBufferStepSystem,
        PortState, PortBuffer, DirtyPort, PortPfcState, PortTimers,
        PortTraceLast, PortCleanup, PortTagList>>({n9});
    auto n11progress = builder.addToGraph<ParallelForNode<Engine,
        progressFinishedSourcesOnePortStepSystem,
        PortState, PortTagList, PortSourceTagList, PortCachedHints,
        PortFinishedSourceList, PortTraceLast, PortOutbox>>({n10a});
    auto n11cleanup = builder.addToGraph<ParallelForNode<Engine,
        cleanupFinishedSourcesOnePortStepSystem,
        PortState, PortCachedHints, PortFinishedSourceList>>({n11progress});
    auto n11drain = builder.addToGraph<ParallelForNode<Engine,
        progressBacklogDrainTimerOnePortStepSystem,
        DirtyPort, PortTimers>>({n11cleanup});
    auto n11flushout = builder.addToGraph<ParallelForNode<Engine,
        flushProgressOutboxStepSystem, SimDriver>>({n11drain});
    auto n11pfctimers = builder.addToGraph<ParallelForNode<Engine,
        progressPfcTimerOnePortStepSystem,
        PortState, PortTimers, IngressTagList, PortDirtyMarkList>>(
            {n11flushout});
    auto n11dirtymarks = builder.addToGraph<ParallelForNode<Engine,
        flushDirtyPortMarksStepSystem, SimDriver>>({n11pfctimers});
    auto n11prepare = builder.addToGraph<ParallelForNode<Engine,
        prepareProgressStepSystem, SimDriver>>({n11dirtymarks});
    auto n11bufferdirty = builder.addToGraph<ParallelForNode<Engine,
        markBufferedPortDirtyOnePortStepSystem,
        PortBuffer, DirtyPort, PortTraceLast>>({n11prepare});
    auto n11finish = builder.addToGraph<ParallelForNode<Engine,
        finishProgressStepSystem, SimDriver>>({n11bufferdirty});
    auto n11bufcleanup = builder.addToGraph<ParallelForNode<Engine,
        materializeTagCleanupOnePortFrameEndStepSystem,
        PortState, PortCleanup, PortIngressUnlinkList,
        PortCompletionList, PortOutbox>>({n11finish});
    auto n12 = builder.addToGraph<ParallelForNode<Engine,
        postBufferStepSystem,
        SimDriver>>({n11bufcleanup});
    final_node = n12;
    }
    }
    }
    }
    }

#ifdef MADRONA_GPU_MODE
    auto recycle_entities = builder.addToGraph<RecycleEntitiesNode>({final_node});
    (void)recycle_entities;
#else
    (void)final_node;
#endif
    initTrace("Sim::setupTasks done", trace_mode_enabled);
}

}
