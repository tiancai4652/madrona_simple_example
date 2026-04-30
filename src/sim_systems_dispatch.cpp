#include "sim.hpp"
#include "sim_debug.hpp"

#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::deliverEvents(Context &ctx)
{
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled =
        traceModeEnabled() && compiledSystemLogEnabled(scope, step);
    int32_t delayed_before = numDelayedEvents;

    if (log_enabled) {
        printSystemBegin(step, now, scope, "deliver_events");
    }

    numInboxArrival = 0;
    numInboxBwUpdate = 0;
    numInboxPfc = 0;

    int32_t due_count = 0;
    while (due_count < numDelayedEvents &&
           delayedEvents[due_count].t <= now + 1e-15) {
        due_count += 1;
    }
    if (workloadStatsEnabled()) {
        stepWorkloadStats.due_events = due_count;
    }

    for (int32_t i = 0; i < due_count; i++) {
        if (delayedEvents[i].type == DelayedEvent::Type::Arrival) {
            const FlowArrivalEv &ev = delayedEvents[i].arrival;
            if (workloadStatsEnabled()) {
                stepWorkloadStats.due_arrival += 1;
            }
            if (numInboxArrival < MAX_EVENTS_PER_STEP) {
                inboxArrival[numInboxArrival++] = ev;
                if (log_enabled) {
                    printSystemDeliverArrival(step, now, ev);
                }
            }
            if (ev.port_id >= 0 && ev.port_id < numPorts) {
                if (workloadStatsEnabled()) {
                    stepDueTouched[ev.port_id] = 1;
                }
                Entity port_e = portEntities[ev.port_id];
                if (port_e != Entity::none()) {
                    PortInbox &inbox = portInboxes[ev.port_id];
                    if (inbox.num_arrival < MAX_PORT_INBOX_ARRIVAL) {
                        inbox.arrivals[inbox.num_arrival++] = ev;
                    }
                }
            }
        } else if (delayedEvents[i].type == DelayedEvent::Type::BwUpdate) {
            const BwUpdateEv &ev = delayedEvents[i].bwupd;
            if (workloadStatsEnabled()) {
                stepWorkloadStats.due_bwupdate += 1;
            }
            if (numInboxBwUpdate < MAX_EVENTS_PER_STEP) {
                inboxBwUpdate[numInboxBwUpdate++] = ev;
                if (log_enabled) {
                    printSystemDeliverBwUpdate(step, now, ev);
                }
            }
            if (ev.port_id >= 0 && ev.port_id < numPorts) {
                if (workloadStatsEnabled()) {
                    stepDueTouched[ev.port_id] = 1;
                }
                Entity port_e = portEntities[ev.port_id];
                if (port_e != Entity::none()) {
                    PortInbox &inbox = portInboxes[ev.port_id];
                    if (inbox.num_bwupd < MAX_PORT_INBOX_BWUPD) {
                        inbox.bwupds[inbox.num_bwupd++] = ev;
                    }
                }
            }
        } else {
            const PfcControlEv &ev = delayedEvents[i].pfcctrl;
            if (workloadStatsEnabled()) {
                stepWorkloadStats.due_pfc += 1;
            }
            if (numInboxPfc < MAX_EVENTS_PER_STEP) {
                inboxPfc[numInboxPfc++] = ev;
                if (log_enabled) {
                    printSystemDeliverPfc(step, now, ev);
                }
            }
            if (ev.target_port_id >= 0 && ev.target_port_id < numPorts) {
                if (workloadStatsEnabled()) {
                    stepDueTouched[ev.target_port_id] = 1;
                }
                Entity port_e = portEntities[ev.target_port_id];
                if (port_e != Entity::none()) {
                    PortInbox &inbox = portInboxes[ev.target_port_id];
                    if (inbox.num_pfc < MAX_PORT_INBOX_PFC) {
                        inbox.pfcs[inbox.num_pfc++] = ev;
                    }
                }
            }
        }
    }

    int32_t remaining = numDelayedEvents - due_count;
    for (int32_t i = 0; i < remaining; i++) {
        delayedEvents[i] = delayedEvents[i + due_count];
    }
    numDelayedEvents = remaining;

    if (workloadStatsEnabled()) {
        int32_t due_ports = 0;
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            due_ports += stepDueTouched[port_id] != 0 ? 1 : 0;
        }
        stepWorkloadStats.due_ports = due_ports;
        stepWorkloadStats.delayed_events_after = numDelayedEvents;
    }

    if (log_enabled) {
        printSystemDeliverSummary(step, now,
            delayed_before,
            numDelayedEvents,
            numInboxArrival,
            numInboxBwUpdate,
            numInboxPfc);
        printSystemEnd(step, now, scope, "deliver_events");
    }
}

void Sim::snapshotDirtyPorts(Context &ctx)
{
    constexpr const char *scope = "emit_pfc";
    uint64_t step = systemLogStep;
    bool log_enabled =
        traceModeEnabled() && compiledSystemLogEnabled(scope, step);
    numLastDirtyPortIDs = 0;
    int32_t cleared_port_count = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortTraceLast &trace = portTraceLasts[port_id];
        if (trace.was_dirty_at_clear != 0) {
            if (numLastDirtyPortIDs < MAX_TOPO_PORTS) {
                lastDirtyPortIDs[numLastDirtyPortIDs++] = port_id;
            }
            cleared_port_count += 1;
            trace.was_dirty_at_clear = 0;
        }
    }

    if (log_enabled) {
        printSystemClearSummary(step, now, cleared_port_count);
    }

    if (workloadStatsEnabled()) {
        stepWorkloadStats.dirty_ports = cleared_port_count;
        stepWorkloadStats.last_dirty_ports = numLastDirtyPortIDs;
    }
}

void Sim::reducePortCachedHints(Context &ctx)
{
    bool reset_drain = false;
    if (cachedDrainPortID >= 0 && cachedDrainPortID < numPorts) {
        Entity cached_port_e = portEntities[cachedDrainPortID];
        if (cached_port_e != Entity::none()) {
            if (portTraceLasts[cachedDrainPortID].was_dirty_at_alloc != 0) {
                reset_drain = true;
            }
        }
    }
    if (reset_drain) {
        cachedNextDrainTime = std::numeric_limits<Time>::max();
        cachedDrainPortID = -1;
    }

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortCachedHints &hints = portCachedHints[port_id];
        if (hints.has_drain_hint != 0) {
            if (hints.drain_hint_t < cachedNextDrainTime) {
                cachedNextDrainTime = hints.drain_hint_t;
                cachedDrainPortID = port_id;
            }
        }
        if (hints.has_finish_hint != 0) {
            if (hints.finish_hint_t < cachedNextFinishTime) {
                cachedNextFinishTime = hints.finish_hint_t;
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
        PortDrainHint &hint = portDrainHints[port_id];
        applyDrainHintOnePort(port_id, hint);
    }
}

void Sim::flushPortTagCleanup(Context &ctx)
{
    int32_t available = MAX_DELAYED_EVENTS - numDelayedEvents;
    if (available < 0) {
        available = 0;
    }
    int32_t batch_count = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortCleanup &cleanup = portCleanups[port_id];
        if (workloadStatsEnabled() && cleanup.num > 0) {
            stepWorkloadStats.cleanup_reqs += cleanup.num;
            if (stepCleanupTouched[port_id] == 0) {
                stepCleanupTouched[port_id] = 1;
                stepWorkloadStats.cleanup_ports += 1;
            }
        }
        for (int32_t i = 0; i < cleanup.num; i++) {
            if (cleanup.tags[i] == Entity::none()) {
                continue;
            }
            DelayedEvent ev {};
            if (destroyTagCollectCleanupEvent(
                    ctx, cleanup.tags[i], cleanup.propagate[i] != 0, now, ev) &&
                batch_count < available) {
                delayedEventScratch[batch_count++] = ev;
            }
        }
        cleanup.num = 0;
    }

    pushDelayedEventsBatch(delayedEventScratch, batch_count);
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
        PortTraceLast &trace = portTraceLasts[port_id];
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

    int32_t available = MAX_DELAYED_EVENTS - numDelayedEvents;
    int32_t batch_count = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortOutbox &outbox = portOutboxes[port_id];
        if (workloadStatsEnabled() && outbox.num_events > 0) {
            stepWorkloadStats.outbox_events += outbox.num_events;
            if (stepOutboxTouched[port_id] == 0) {
                stepOutboxTouched[port_id] = 1;
                stepWorkloadStats.outbox_ports += 1;
            }
        }
        int32_t remaining = available - batch_count;
        int32_t take = outbox.num_events;
        if (take > remaining) {
            take = remaining;
        }
        for (int32_t i = 0; i < take; i++) {
            delayedEventScratch[batch_count++] = outbox.events[i];
        }
        outbox.num_events = 0;
    }

    pushDelayedEventsBatch(delayedEventScratch, batch_count);
}

void Sim::flushPortPfcTimers(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortPfcState &state = portPfcStates[port_id];
        applyPfcTimerOnePort(port_id, state);
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
            PortTraceLast &trace = portTraceLasts[port_id];
            checked += trace.pfc_detect_checked;
            emitted += trace.pfc_detect_emitted;
        }
    }

    printSystemPfcDetectSummary(step, now, checked, emitted,
        countActivePfcPauseTimers(), countActivePfcResumeTimers());
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
        PortTraceLast &trace = portTraceLasts[port_id];
        dirty_count += trace.emit_is_dirty;
        arrival_count += trace.emit_arrival_count;
        bwupdate_count += trace.emit_bwupdate_count;
    }

    printSystemEmitSummary(step, now, dirty_count,
        arrival_count, bwupdate_count);
}

}
