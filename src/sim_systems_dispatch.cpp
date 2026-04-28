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

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortInbox &inbox = portInboxes[port_id];
        inbox.num_arrival = 0;
        inbox.num_bwupd = 0;
        inbox.num_pfc = 0;
        portCreateLists[port_id].num = 0;
        portCompletionLists[port_id].num = 0;
    }

    int32_t write_idx = 0;
    for (int32_t i = 0; i < numDelayedEvents; i++) {
        if (delayedEvents[i].t <= now + 1e-15) {
            if (delayedEvents[i].type == DelayedEvent::Type::Arrival) {
                const FlowArrivalEv &ev = delayedEvents[i].arrival;
                if (numInboxArrival < MAX_EVENTS_PER_STEP) {
                    inboxArrival[numInboxArrival++] = ev;
                    if (log_enabled) {
                        printSystemDeliverArrival(step, now, ev);
                    }
                }
                if (ev.port_id >= 0 && ev.port_id < numPorts) {
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
                if (numInboxBwUpdate < MAX_EVENTS_PER_STEP) {
                    inboxBwUpdate[numInboxBwUpdate++] = ev;
                    if (log_enabled) {
                        printSystemDeliverBwUpdate(step, now, ev);
                    }
                }
                if (ev.port_id >= 0 && ev.port_id < numPorts) {
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
                if (numInboxPfc < MAX_EVENTS_PER_STEP) {
                    inboxPfc[numInboxPfc++] = ev;
                    if (log_enabled) {
                        printSystemDeliverPfc(step, now, ev);
                    }
                }
                if (ev.target_port_id >= 0 && ev.target_port_id < numPorts) {
                    Entity port_e = portEntities[ev.target_port_id];
                    if (port_e != Entity::none()) {
                        PortInbox &inbox = portInboxes[ev.target_port_id];
                        if (inbox.num_pfc < MAX_PORT_INBOX_PFC) {
                            inbox.pfcs[inbox.num_pfc++] = ev;
                        }
                    }
                }
            }
        } else {
            delayedEvents[write_idx++] = delayedEvents[i];
        }
    }
    numDelayedEvents = write_idx;

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
        if (hint.want_clear != 0) {
            clearBacklogDrainTimer(port_id);
        }
    }
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortDrainHint &hint = portDrainHints[port_id];
        if (hint.want_set != 0) {
            int32_t idx = findBacklogDrainTimerIndex(port_id);
            if (idx < 0 || hint.set_t < backlogDrainTimers[idx]) {
                setBacklogDrainTimer(port_id, hint.set_t);
            }
        }
    }
}

void Sim::flushPortTagCleanup(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortCleanup &cleanup = portCleanups[port_id];
        for (int32_t i = 0; i < cleanup.num; i++) {
            if (cleanup.tags[i] == Entity::none()) {
                continue;
            }
            destroyTag(ctx, cleanup.tags[i], cleanup.propagate[i] != 0, now);
        }
        cleanup.num = 0;
    }
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
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortOutbox &outbox = portOutboxes[port_id];
        for (int32_t i = 0; i < outbox.num_events; i++) {
            pushDelayedEvent(outbox.events[i]);
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
        PortPfcState &state = portPfcStates[port_id];
        if (state.want_clear_pause != 0) {
            clearPfcPauseTimer(port_id);
        }
        if (state.want_clear_resume != 0) {
            clearPfcResumeTimer(port_id);
        }
        if (state.want_set_pause != 0) {
            setPfcPauseTimer(port_id, state.set_pause_t);
        }
        if (state.want_set_resume != 0) {
            setPfcResumeTimer(port_id, state.set_resume_t);
        }
        state.want_clear_pause = 0;
        state.want_clear_resume = 0;
        state.want_set_pause = 0;
        state.want_set_resume = 0;
        state.set_pause_t = 0.0;
        state.set_resume_t = 0.0;
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
        numPfcPauseTimers, numPfcResumeTimers);
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
