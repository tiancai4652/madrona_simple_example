#include "sim.hpp"
#include "sim_debug.hpp"

#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

inline void compactPortDelayedQueue(PortDelayedQueue &queue)
{
    if (queue.head <= 0) {
        return;
    }

    if (queue.count <= 0) {
        queue.head = 0;
        return;
    }

    for (int32_t i = 0; i < queue.count; i++) {
        queue.events[i] = queue.events[queue.head + i];
    }
    queue.head = 0;
}

}

void Sim::deliverEventsOnePort(Context &ctx,
                               int32_t port_id,
                               PortDelayedQueue &queue,
                               PortInbox &inbox,
                               PortTraceLast &trace)
{
    (void)ctx;
    (void)port_id;
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled =
        traceModeEnabled() && compiledSystemLogEnabled(scope, step);
    trace.deliver_arrival_count = 0;
    trace.deliver_bwupdate_count = 0;
    trace.deliver_pfc_count = 0;
    trace.deliver_pfc_dropped = 0;

    int32_t due_count = 0;
    while (due_count < queue.count &&
           queue.events[queue.head + due_count].t <= now + 1e-15) {
        due_count += 1;
    }

    for (int32_t i = 0; i < due_count; i++) {
        const DelayedEvent &dev = queue.events[queue.head + i];
        if (dev.type == DelayedEvent::Type::Arrival) {
            const FlowArrivalEv &ev = dev.arrival;
            trace.deliver_arrival_count += 1;
            if (log_enabled) {
                printSystemDeliverArrival(step, now, ev);
            }
            if (inbox.num_arrival < MAX_PORT_INBOX_ARRIVAL) {
                inbox.arrivals[inbox.num_arrival++] = ev;
            }
        } else if (dev.type == DelayedEvent::Type::BwUpdate) {
            const BwUpdateEv &ev = dev.bwupd;
            trace.deliver_bwupdate_count += 1;
            if (log_enabled) {
                printSystemDeliverBwUpdate(step, now, ev);
            }
            if (inbox.num_bwupd < MAX_PORT_INBOX_BWUPD) {
                inbox.bwupds[inbox.num_bwupd++] = ev;
            }
        } else {
            const PfcControlEv &ev = dev.pfcctrl;
            trace.deliver_pfc_count += 1;
            if (log_enabled) {
                printSystemDeliverPfc(step, now, ev);
            }
            if (inbox.num_pfc < MAX_PORT_INBOX_PFC) {
                inbox.pfcs[inbox.num_pfc++] = ev;
            } else {
                trace.deliver_pfc_dropped += 1;
            }
        }
    }

    queue.head += due_count;
    queue.count -= due_count;
    if (queue.count == 0) {
        queue.head = 0;
    } else if (queue.head > (MAX_PORT_DELAYED_EVENTS / 4) &&
               queue.head >= queue.count) {
        compactPortDelayedQueue(queue);
    }
}

void Sim::finishDeliverEvents(Context &ctx)
{
    (void)ctx;
    SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled =
        traceModeEnabled() && compiledSystemLogEnabled(scope, step);
    int32_t delayed_before = runtime.numDelayedEvents;
    int32_t delayed_after = 0;
    int32_t inbox_arrival_count = 0;
    int32_t inbox_bwupdate_count = 0;
    int32_t inbox_pfc_count = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        delayed_after += ctx.get<PortDelayedQueue>(port_e).count;
        PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
        inbox_arrival_count += trace.deliver_arrival_count;
        inbox_bwupdate_count += trace.deliver_bwupdate_count;
        inbox_pfc_count += trace.deliver_pfc_count;
    }

    runtime.numDelayedEvents = delayed_after;

    if (log_enabled) {
        printSystemDeliverSummary(step, now,
            delayed_before,
            delayed_after,
            inbox_arrival_count,
            inbox_bwupdate_count,
            inbox_pfc_count);
        printSystemEnd(step, now, scope, "deliver_events");
    }
}

MADRONA_NO_INLINE void Sim::snapshotDirtyOnePort(
    int32_t,
    const PortDelayedQueue &queue,
    const PortTimers &timers,
    PortTraceLast &trace) const
{
    trace.clear_has_delayed_gap = 0;
    trace.clear_delayed_gap = 0.0;
    trace.clear_has_backlog_gap = 0;
    trace.clear_backlog_gap = 0.0;
    trace.clear_has_pfc_pause_gap = 0;
    trace.clear_pfc_pause_gap = 0.0;
    trace.clear_has_pfc_resume_gap = 0;
    trace.clear_pfc_resume_gap = 0.0;

    if (queue.count > 0) {
        Time gap = queue.events[queue.head].t - now;
        if (gap > 1e-15) {
            trace.clear_has_delayed_gap = 1;
            trace.clear_delayed_gap = gap;
        }
    }

    if (timers.backlog_drain > 1e-15 &&
        timerIsActive(timers.backlog_drain)) {
        trace.clear_has_backlog_gap = 1;
        trace.clear_backlog_gap = timers.backlog_drain;
    }

    if (enablePfc != 0) {
        if (timers.pfc_pause > 1e-9 &&
            timerIsActive(timers.pfc_pause)) {
            trace.clear_has_pfc_pause_gap = 1;
            trace.clear_pfc_pause_gap = timers.pfc_pause;
        }

        if (timers.pfc_resume > 1e-9 &&
            timerIsActive(timers.pfc_resume)) {
            trace.clear_has_pfc_resume_gap = 1;
            trace.clear_pfc_resume_gap = timers.pfc_resume;
        }
    }
}

void Sim::snapshotDirtyPorts(Context &ctx)
{
    SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    constexpr const char *scope = "emit_pfc";
    uint64_t step = systemLogStep;
    bool log_enabled =
        traceModeEnabled() && compiledSystemLogEnabled(scope, step);
    int32_t cleared_port_count = 0;
    runtime.cachedNextDelayedGap = timerInactiveSentinel();
    runtime.cachedNextBacklogGap = timerInactiveSentinel();
    runtime.cachedNextPfcPauseGap = timerInactiveSentinel();
    runtime.cachedNextPfcResumeGap = timerInactiveSentinel();

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
        if (trace.was_dirty_at_clear != 0) {
            cleared_port_count += 1;
        }

        if (trace.clear_has_delayed_gap != 0 &&
            trace.clear_delayed_gap < runtime.cachedNextDelayedGap) {
            runtime.cachedNextDelayedGap = trace.clear_delayed_gap;
        }

        if (trace.clear_has_backlog_gap != 0 &&
            trace.clear_backlog_gap < runtime.cachedNextBacklogGap) {
            runtime.cachedNextBacklogGap = trace.clear_backlog_gap;
        }

        if (enablePfc != 0) {
            if (trace.clear_has_pfc_pause_gap != 0 &&
                trace.clear_pfc_pause_gap < runtime.cachedNextPfcPauseGap) {
                runtime.cachedNextPfcPauseGap = trace.clear_pfc_pause_gap;
            }

            if (trace.clear_has_pfc_resume_gap != 0 &&
                trace.clear_pfc_resume_gap < runtime.cachedNextPfcResumeGap) {
                runtime.cachedNextPfcResumeGap =
                    trace.clear_pfc_resume_gap;
            }
        }
    }

    if (log_enabled) {
        printSystemClearSummary(step, now, cleared_port_count);
    }

}

void Sim::reducePortCachedHints(Context &ctx)
{
    SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    bool reset_drain = false;
    if (runtime.cachedDrainPortID >= 0 &&
        runtime.cachedDrainPortID < numPorts) {
        Entity cached_port_e = portEntities[runtime.cachedDrainPortID];
        if (cached_port_e != Entity::none()) {
            if (ctx.get<PortTraceLast>(
                    cached_port_e).was_dirty_at_alloc != 0) {
                reset_drain = true;
            }
        }
    }
    if (reset_drain) {
        runtime.cachedNextDrainTime = std::numeric_limits<Time>::max();
        runtime.cachedDrainPortID = -1;
    }

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortCachedHints &hints = ctx.get<PortCachedHints>(port_e);
        PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
        if (trace.was_dirty_at_alloc != 0) {
            if (hints.has_finish_hint != 0) {
                if (hints.finish_hint_t < runtime.cachedNextFinishTime) {
                    runtime.cachedNextFinishTime = hints.finish_hint_t;
                }
            }
        }
        if (hints.has_drain_hint != 0) {
            if (hints.drain_hint_t < runtime.cachedNextDrainTime) {
                runtime.cachedNextDrainTime = hints.drain_hint_t;
                runtime.cachedDrainPortID = port_id;
            }
        }
    }
}

void Sim::flushPortDrainHints(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortDrainHint &hint = ctx.get<PortDrainHint>(port_e);
        PortTimers &timers = ctx.get<PortTimers>(port_e);
        applyDrainHintOnePort(port_id, hint, timers);
    }
}

void Sim::flushPortTagCleanup(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortCleanup &cleanup = ctx.get<PortCleanup>(port_e);
        for (int32_t i = 0; i < cleanup.num; i++) {
            if (cleanup.tags[i] == Entity::none()) {
                continue;
            }
            DelayedEvent ev {};
            if (destroyTagCollectCleanupEvent(
                    ctx, cleanup.tags[i], cleanup.propagate[i] != 0, now, ev)) {
                pushDelayedEvent(ctx, ev);
            }
        }
        cleanup.num = 0;
    }
}

MADRONA_NO_INLINE void Sim::materializeTagCleanupOnePort(
    Context &ctx,
    int32_t,
    PortCleanup &cleanup,
    PortIngressUnlinkList &ingress_unlinks,
    PortCompletionList &completions,
    PortOutbox &outbox,
    Time logical_now)
{
    for (int32_t i = 0; i < cleanup.num; i++) {
        if (cleanup.tags[i] == Entity::none()) {
            continue;
        }

        destroyTagMaterializeOnePort(ctx, cleanup.tags[i],
            cleanup.propagate[i] != 0, logical_now,
            ingress_unlinks, completions, outbox);
    }

    cleanup.num = 0;
}

void Sim::logAllocTraces(Context &ctx)
{
    if (!traceModeEnabled()) {
        return;
    }

    constexpr const char *scope = "alloc";
    uint64_t step = systemLogStep;
    bool log_enabled = compiledSystemLogEnabled(scope, step);
    if (!log_enabled) {
        return;
    }

    int32_t num_dirty = 0;
    int32_t processed = 0;
    int32_t dirty_tag_count = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
        if (trace.was_dirty_at_alloc == 0) {
            continue;
        }
        num_dirty += 1;
        if (trace.alloc_num_tags > 0) {
            processed += 1;
            dirty_tag_count += trace.alloc_num_tags;
        }
        if (log_enabled && trace.has_alloc_trace != 0) {
            printSystemAllocPort(step, now, port_id,
                trace.alloc_port_bw,
                trace.alloc_num_tags,
                trace.alloc_num_live,
                trace.alloc_sum_in,
                trace.alloc_sum_out,
                qosMode,
                trace.alloc_is_dest_only,
                trace.alloc_has_buffer);
        }
    }
    printSystemAllocSummary(step, now, num_dirty, processed, dirty_tag_count);
}

void Sim::flushPortOutbox(Context &ctx)
{
    (void)ctx;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortOutbox &outbox = ctx.get<PortOutbox>(port_e);
        for (int32_t i = 0; i < outbox.num_events; i++) {
            pushDelayedEvent(ctx, outbox.events[i]);
        }
        outbox.num_events = 0;
    }
}

void Sim::flushPortPfcTimers(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortPfcState &state = ctx.get<PortPfcState>(port_e);
        PortTimers &timers = ctx.get<PortTimers>(port_e);
        applyPfcTimerOnePort(port_id, state, timers);
    }
}

void Sim::logPfcDetectTraces(Context &ctx)
{
    if (!traceModeEnabled()) {
        return;
    }

    constexpr const char *scope = "emit_pfc";
    uint64_t step = systemLogStep;
    if (!compiledSystemLogEnabled(scope, step)) {
        return;
    }

    int32_t checked = 0;
    int32_t emitted = 0;
    if (enablePfc != 0) {
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity port_e = portEntities[port_id];
            if (port_e == Entity::none()) {
                continue;
            }
            PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
            checked += trace.pfc_detect_checked;
            emitted += trace.pfc_detect_emitted;
        }
    }

    printSystemPfcDetectSummary(step, now, checked, emitted,
        countActivePfcPauseTimers(ctx), countActivePfcResumeTimers(ctx));
}

void Sim::logPfcDebugTraces(Context &ctx)
{
    if (!traceModeEnabled()) {
        return;
    }

    constexpr const char *scope = "pfc_debug";
    uint64_t step = systemLogStep;
    if (!compiledSystemLogEnabled(scope, step)) {
        return;
    }

    int32_t detect_checked = 0;
    int32_t detect_emitted = 0;
    int32_t detect_dropped = 0;
    int32_t delivered = 0;
    int32_t deliver_dropped = 0;
    int32_t applied = 0;
    int32_t skipped = 0;
    int32_t active_pause_ports = 0;
    int32_t active_paused_ports = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
        detect_checked += trace.pfc_detect_checked;
        detect_emitted += trace.pfc_detect_emitted;
        detect_dropped += trace.pfc_detect_dropped;
        delivered += trace.deliver_pfc_count;
        deliver_dropped += trace.deliver_pfc_dropped;
        applied += trace.pfc_applied;
        skipped += trace.pfc_skipped;

        const PortPfcState &pfc = ctx.get<PortPfcState>(port_e);
        bool has_pause_active = false;
        bool has_paused = false;
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            has_pause_active = has_pause_active || pfc.pause_active[pri] != 0;
            has_paused = has_paused || pfc.paused[pri] != 0;
        }
        active_pause_ports += has_pause_active ? 1 : 0;
        active_paused_ports += has_paused ? 1 : 0;
    }

    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    printSystemPfcDebugSummary(step, now,
        detect_checked,
        detect_emitted,
        detect_dropped,
        delivered,
        deliver_dropped,
        applied,
        skipped,
        runtime.delayedDropCount,
        runtime.delayedPfcDropCount,
        active_pause_ports,
        active_paused_ports);
}

void Sim::logEmitTraces(Context &ctx)
{
    if (!traceModeEnabled()) {
        return;
    }

    constexpr const char *scope = "emit_pfc";
    uint64_t step = systemLogStep;
    if (!compiledSystemLogEnabled(scope, step)) {
        return;
    }

    int32_t dirty_count = 0;
    int32_t arrival_count = 0;
    int32_t bwupdate_count = 0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
        dirty_count += trace.emit_is_dirty;
        arrival_count += trace.emit_arrival_count;
        bwupdate_count += trace.emit_bwupdate_count;
    }

    printSystemEmitSummary(step, now, dirty_count,
        arrival_count, bwupdate_count);
}

}
