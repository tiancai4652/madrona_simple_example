#include "sim.hpp"
#include "sim_debug.hpp"
#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#endif

using namespace madrona;
using namespace madrona::math;

namespace madsimple {
// Emit a one-shot init trace. Suppressed when init_log_print_enabled is on,
// or when FCT-only perf mode is enabled, so the [INIT] log dump consumed by
// check/run_parity.py stays free of any extra `[init-trace]` lines that would
// shift line numbers and break diffs. On GPU, only thread 0 prints to avoid
// flooding the CUDA printf buffer while initTasks builds the task graph on
// device.
static inline void initTrace(const char *msg, bool trace_mode_enabled)
{
    if (!trace_mode_enabled) {
        return;
    }

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

// Compile-time diagnostic switch for bisecting final NVVM link pressure.
// Stage meanings in setupTasks():
//   4 = through ingress chain
//   5 = through alloc
//   6 = through pfc
//   7 = through emit
//   9 = through clear/chooseDT
//  12 = full graph
constexpr int kTaskgraphStageLimit = 12;

MADRONA_NO_INLINE void scheduleStepSystem(Engine &ctx, SimDriver &driver)
{
    Sim &sim = ctx.data();
    if (sim.traceModeEnabled()) {
        sim.systemLogStep += 1;
    }
    driver.tick += 1;
    // [step-trace] SimDriver 只有 1 entity, ParallelForNode 在 megakernel
    // 里只会派 1 个 thread 跑这个 system，但 thread index 不固定，所以
    // 不能用 threadIdx.x == 0 过滤。只用 tick<=3 限频。
#ifdef MADRONA_GPU_MODE
    if constexpr (step_trace_compiled_in) {
        if (sim.traceModeEnabled() && driver.tick <= 3) {
            int32_t t = driver.tick;
            int32_t pend_before = sim.numPendingFlows;
            float now_f = (float)sim.now;
            mwGPU::HostPrint::log(
                "[step-trace] scheduleStepSystem tick=%d pend_before=%d now=%f\n",
                t, pend_before, now_f);
        }
    }
#endif
    sim.schedulePendingFlows();
#ifdef MADRONA_GPU_MODE
    if constexpr (step_trace_compiled_in) {
        if (sim.traceModeEnabled() && driver.tick <= 3) {
            int32_t t = driver.tick;
            int32_t pend_after = sim.numPendingFlows;
            int32_t delayed_after = sim.numDelayedEvents;
            mwGPU::HostPrint::log(
                "[step-trace] scheduleStepSystem AFTER tick=%d pend=%d delayed=%d\n",
                t, pend_after, delayed_after);
        }
    }
#endif
}

MADRONA_NO_INLINE void deliverStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.deliverEvents(ctx);
}

MADRONA_NO_INLINE void resetIngressPortStateStepSystem(
    Engine &ctx,
    PortState &port_state)
{
    Sim &sim = ctx.data();
    int32_t port_id = port_state.port_id;
    PortInbox &inbox = sim.portInboxes[port_id];
    inbox.num_arrival = 0;
    inbox.num_bwupd = 0;
    inbox.num_pfc = 0;
    sim.portCreateLists[port_id].num = 0;
    sim.portCompletionLists[port_id].num = 0;
}

// Phase E: per-Port ingress-chain workers. Each operates on a single
// port's inbox + local components only; all cross-port effects are
// deferred to flushTagCreate / flushFlowCompletion / flushPortOutbox /
// flushTagCleanup singletons that follow.
MADRONA_NO_INLINE void pfcPropagateOnePortStepSystem(
    Engine &ctx,
    PortState &port_state)
{
    Sim &sim = ctx.data();
    DirtyPort &dirty = sim.portDirtyStates[port_state.port_id];
    PortPfcState &pfc_state = sim.portPfcStates[port_state.port_id];
    PortTraceLast &trace = sim.portTraceLasts[port_state.port_id];
    PortInbox &inbox = sim.portInboxes[port_state.port_id];
    sim.pfcPropagateOnePort(ctx, port_state.port_id, port_state, pfc_state,
        dirty, inbox, trace);
}

MADRONA_NO_INLINE void arrivalOnePortStepSystem(
    Engine &ctx,
    PortState &port_state)
{
    Sim &sim = ctx.data();
    DirtyPort &dirty = sim.portDirtyStates[port_state.port_id];
    PortTraceLast &trace = sim.portTraceLasts[port_state.port_id];
    PortInbox &inbox = sim.portInboxes[port_state.port_id];
    PortTagList &tag_list = sim.portTagLists[port_state.port_id];
    PortCreateList &create_list = sim.portCreateLists[port_state.port_id];
    sim.flowArrivalOnePort(ctx, port_state.port_id, port_state, dirty, inbox,
        tag_list, create_list, trace);
}

MADRONA_NO_INLINE void bwUpdateOnePortStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf)
{
    Sim &sim = ctx.data();
    DirtyPort &dirty = sim.portDirtyStates[port_state.port_id];
    PortTraceLast &trace = sim.portTraceLasts[port_state.port_id];
    PortInbox &inbox = sim.portInboxes[port_state.port_id];
    PortTagList &tag_list = sim.portTagLists[port_state.port_id];
    PortCreateList &create_list = sim.portCreateLists[port_state.port_id];
    PortCleanup &cleanup = sim.portCleanups[port_state.port_id];
    PortOutbox &outbox = sim.portOutboxes[port_state.port_id];
    PortCompletionList &completions =
        sim.portCompletionLists[port_state.port_id];
    sim.bwUpdateOnePort(ctx, port_state.port_id, port_state, port_buf, dirty,
        inbox, tag_list, create_list, cleanup, outbox, completions, trace);
}

MADRONA_NO_INLINE void flushTagCreateStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushTagCreate(ctx);
}

MADRONA_NO_INLINE void postIngressStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
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
    PortBuffer &port_buf)
{
    Sim &sim = ctx.data();
    DirtyPort &dirty = sim.portDirtyStates[port_state.port_id];
    PortPfcConfig &pfc_cfg = sim.portPfcConfigs[port_state.port_id];
    PortPfcState &pfc_state = sim.portPfcStates[port_state.port_id];
    PortCachedHints &hints = sim.portCachedHints[port_state.port_id];
    PortDrainHint &drain_hint = sim.portDrainHints[port_state.port_id];
    PortTraceLast &trace = sim.portTraceLasts[port_state.port_id];
    PortCleanup &cleanup = sim.portCleanups[port_state.port_id];
    PortTagList &tag_list = sim.portTagLists[port_state.port_id];
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
    PortState &port_state)
{
    Sim &sim = ctx.data();
    PortDrainHint &hint = sim.portDrainHints[port_state.port_id];
    sim.applyDrainHintOnePort(port_state.port_id, hint);
}

MADRONA_NO_INLINE void postAllocStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortTagCleanup(ctx);
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
    PortBuffer &port_buf)
{
    Sim &sim = ctx.data();
    DirtyPort &dirty = sim.portDirtyStates[port_state.port_id];
    PortPfcConfig &pfc_cfg = sim.portPfcConfigs[port_state.port_id];
    PortPfcState &pfc_state = sim.portPfcStates[port_state.port_id];
    PortTraceLast &trace = sim.portTraceLasts[port_state.port_id];
    PortOutbox &outbox = sim.portOutboxes[port_state.port_id];
    PortTagList &tag_list = sim.portTagLists[port_state.port_id];
    sim.pfcDetectOnePort(ctx, port_state.port_id, port_state, port_buf,
        dirty, pfc_cfg, pfc_state, outbox, trace, tag_list);
}

// Phase C: per-Port downstream emit. Pushes Arrival/BwUpdate events into
// this port's own PortOutbox; flushPortOutbox later appends them to
// Sim::delayedEvents in port_id ascending order.
MADRONA_NO_INLINE void emitOnePortStepSystem(
    Engine &ctx,
    PortState &port_state)
{
    Sim &sim = ctx.data();
    DirtyPort &dirty = sim.portDirtyStates[port_state.port_id];
    PortTraceLast &trace = sim.portTraceLasts[port_state.port_id];
    PortOutbox &outbox = sim.portOutboxes[port_state.port_id];
    PortTagList &tag_list = sim.portTagLists[port_state.port_id];
    sim.emitOnePort(ctx, port_state.port_id, port_state, dirty, outbox, trace,
        tag_list);
}

// Phase C singletons.
MADRONA_NO_INLINE void applyPfcTimerOnePortStepSystem(
    Engine &ctx,
    PortState &port_state)
{
    Sim &sim = ctx.data();
    PortPfcState &state = sim.portPfcStates[port_state.port_id];
    sim.applyPfcTimerOnePort(port_state.port_id, state);
}

MADRONA_NO_INLINE void postPfcStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flushPortOutbox(ctx);
    if (sim.traceModeEnabled()) {
        sim.logPfcDetectTraces(ctx);
    }
}

MADRONA_NO_INLINE void postEmitStepSystem(Engine &ctx, SimDriver &)
{
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
                                               PortState &port_state)
{
    Sim &sim = ctx.data();
    DirtyPort &dirty = sim.portDirtyStates[port_state.port_id];
    PortTraceLast &trace = sim.portTraceLasts[port_state.port_id];
    trace.was_dirty_at_clear = (dirty.isDirty != 0) ? 1 : 0;
    dirty.isDirty = 0;
}

// Singleton driven by SimDriver: rebuilds lastDirtyPortIDs in port_id
// ascending order, then computes nextDT for the upcoming buffer step. These
// operations were adjacent singleton nodes already; keeping them together
// reduces taskgraph / userEntry count without changing ordering.
MADRONA_NO_INLINE void postClearStepSystem(Engine &ctx, SimDriver &driver)
{
    Sim &sim = ctx.data();
    (void)driver;
    sim.snapshotDirtyPorts(ctx);
    sim.nextDT = sim.chooseDT();
    if (sim.nextDT < 1e-9) {
        sim.nextDT = 0.001;
    }
#ifdef MADRONA_GPU_MODE
    if constexpr (step_trace_compiled_in) {
        if (sim.traceModeEnabled() && driver.tick <= 3) {
            int32_t t = driver.tick;
            float dt_f = (float)sim.nextDT;
            float now_f = (float)sim.now;
            int32_t pend = sim.numPendingFlows;
            int32_t delayed = sim.numDelayedEvents;
            mwGPU::HostPrint::log(
                "[step-trace] chooseDTStepSystem tick=%d nextDT=%f now=%f pend=%d delayed=%d\n",
                t, dt_f, now_f, pend, delayed);
        }
    }
#endif
}

// Phase B.3: per-Port ParallelForNode that advances a single port's
// buffer state for dt = sim.nextDT. Mirrors allocOnePortStepSystem in
// that each port only touches its own components plus read-only
// topology/tagIndex scans; cross-port mutation (destroyTag) is deferred
// into PortCleanup and flushed by the singleton that follows.
MADRONA_NO_INLINE void advanceOnePortBufferStepSystem(
    Engine &ctx,
    PortState &port_state,
    PortBuffer &port_buf)
{
    Sim &sim = ctx.data();
    DirtyPort &dirty = sim.portDirtyStates[port_state.port_id];
    PortPfcState &pfc_state = sim.portPfcStates[port_state.port_id];
    PortTraceLast &trace = sim.portTraceLasts[port_state.port_id];
    PortCleanup &cleanup = sim.portCleanups[port_state.port_id];
    PortTagList &tag_list = sim.portTagLists[port_state.port_id];
    sim.advanceOnePortBuffer(ctx, port_state.port_id, sim.nextDT,
        port_state, port_buf, dirty, pfc_state, cleanup, trace, tag_list);
}

MADRONA_NO_INLINE void postBufferStepSystem(Engine &ctx, SimDriver &driver)
{
    Sim &sim = ctx.data();
    sim.flushBufferTagCleanup(ctx);
    if (sim.traceModeEnabled()) {
        sim.logBufferTraces(ctx);
    }
#ifdef MADRONA_GPU_MODE
    if constexpr (step_trace_compiled_in) {
        if (sim.traceModeEnabled() && driver.tick <= 3) {
            int32_t t = driver.tick;
            float dt_f = (float)sim.nextDT;
            float now_f = (float)sim.now;
            mwGPU::HostPrint::log(
                "[step-trace] flowProgressStepSystem BEFORE tick=%d nextDT=%f now=%f\n",
                t, dt_f, now_f);
        }
    }
#endif
    sim.flowProgressAndCleanupSystem(ctx, sim.nextDT);
    sim.now += sim.nextDT;
#ifdef MADRONA_GPU_MODE
    if constexpr (step_trace_compiled_in) {
        if (sim.traceModeEnabled() && driver.tick <= 3) {
            int32_t t = driver.tick;
            float now_f = (float)sim.now;
            mwGPU::HostPrint::log(
                "[step-trace] flowProgressStepSystem AFTER tick=%d now=%f\n",
                t, now_f);
        }
    }
#endif
    SimStats &stats = ctx.singleton<SimStats>();
    FlowCompletionBuf &buf = ctx.singleton<FlowCompletionBuf>();
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

    auto n0 = builder.addToGraph<ParallelForNode<Engine,
        scheduleStepSystem, SimDriver>>({});
    auto n1reset = builder.addToGraph<ParallelForNode<Engine,
        resetIngressPortStateStepSystem, PortState>>({n0});
    auto n1 = builder.addToGraph<ParallelForNode<Engine,
        deliverStepSystem, SimDriver>>({n1reset});
    // Phase E: per-Port ingress chain. resetIngressPortStateStepSystem
    // has already cleared each port's inbox / create / completion
    // scratch, and deliverEvents (n1) has dispatched each due event into
    // the target port's PortInbox. We run pfcPropagate → arrival → flushTagCreate → bwUpdate →
    // flushTagCreate → flushTagCleanup → flushFlowCompletion →
    // flushPortOutbox → logIngressChain, matching the effective order
    // the legacy singleton path produced.
    auto n2pfc = builder.addToGraph<ParallelForNode<Engine,
        pfcPropagateOnePortStepSystem,
        PortState>>({n1});
    auto n2arr = builder.addToGraph<ParallelForNode<Engine,
        arrivalOnePortStepSystem,
        PortState>>({n2pfc});
    auto n2createA = builder.addToGraph<ParallelForNode<Engine,
        flushTagCreateStepSystem, SimDriver>>({n2arr});
    auto n2bw = builder.addToGraph<ParallelForNode<Engine,
        bwUpdateOnePortStepSystem,
        PortState, PortBuffer>>({n2createA});
    auto n2createB = builder.addToGraph<ParallelForNode<Engine,
        flushTagCreateStepSystem, SimDriver>>({n2bw});
    auto n2cleanup = builder.addToGraph<ParallelForNode<Engine,
        flushTagCleanupStepSystem, SimDriver>>({n2createB});
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
    PortState, PortBuffer>>({n4});
    auto n5reduce = builder.addToGraph<ParallelForNode<Engine,
        reducePortCachedHintsStepSystem, SimDriver>>({n5a});
    auto n5drain = builder.addToGraph<ParallelForNode<Engine,
        applyDrainHintOnePortStepSystem, PortState>>({n5reduce});
    auto n5 = builder.addToGraph<ParallelForNode<Engine,
        postAllocStepSystem, SimDriver>>({n5drain});
    final_node = n5;
    // Phase C: per-Port pfcDetect fan-out, followed by a per-Port timer
    // apply pass, then the singleton outbox flush / summary log. PFC
    // pause/resume timers are now port-indexed, so applying each port's
    // deferred want_* state is race-free. We still flush the PFC-phase
    // PortOutbox into Sim::delayedEvents BEFORE the emit per-Port nodes
    // write their own Arrival/BwUpdate events into the same outbox, so
    // delayedEvents ordering remains legacy-compatible.
    if constexpr (kTaskgraphStageLimit >= 6) {
    auto n6a = builder.addToGraph<ParallelForNode<Engine,
        pfcDetectOnePortStepSystem,
        PortState, PortBuffer>>({n5});
    auto n6timers = builder.addToGraph<ParallelForNode<Engine,
        applyPfcTimerOnePortStepSystem, PortState>>({n6a});
    auto n6 = builder.addToGraph<ParallelForNode<Engine,
        postPfcStepSystem, SimDriver>>({n6timers});
    final_node = n6;
    // Phase C: per-Port downstream emit fan-out, followed by an outbox
    // flush and the emit summary log singleton.
    if constexpr (kTaskgraphStageLimit >= 7) {
    auto n7a = builder.addToGraph<ParallelForNode<Engine,
        emitOnePortStepSystem,
        PortState>>({n6});
    auto n7 = builder.addToGraph<ParallelForNode<Engine,
        postEmitStepSystem, SimDriver>>({n7a});
    final_node = n7;
    // Phase B.1: per-Port clearDirtyOnePortSystem fans out over every Port
    // entity; the follow-up SimDriver singleton snapshotDirtyStepSystem
    // collapses the per-port snapshots into lastDirtyPortIDs in
    // port_id ascending order and emits the clear-summary log.
    if constexpr (kTaskgraphStageLimit >= 9) {
    auto n8a = builder.addToGraph<ParallelForNode<Engine,
        clearDirtyOnePortSystem, PortState>>({n7});
    auto n9 = builder.addToGraph<ParallelForNode<Engine,
        postClearStepSystem, SimDriver>>({n8a});
    final_node = n9;
    // Phase B.3: per-Port fan-out of the buffer advance phase, followed by
    // two SimDriver singletons. flushBufferTagCleanup replays deferred
    // destroyTag in port_id ascending order; logBufferTraces sums the
    // per-port buffer summary fields before we advance `now`.
    if constexpr (kTaskgraphStageLimit >= 12) {
    auto n10a = builder.addToGraph<ParallelForNode<Engine,
        advanceOnePortBufferStepSystem,
        PortState, PortBuffer>>({n9});
    auto n12 = builder.addToGraph<ParallelForNode<Engine,
        postBufferStepSystem,
        SimDriver>>({n10a});
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
