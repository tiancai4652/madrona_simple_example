#include "sim.hpp"
#include "sim_debug.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

// Helper: find a tag belonging to this port by flow_id, using the
// PortTagList mirror. O(tag_list.count) instead of O(numTagIndexEntries).
inline Entity findTagInPortList(Context &ctx,
                                const PortTagList &tag_list,
                                FlowId flow_id)
{
    for (int32_t i = 0; i < tag_list.count; i++) {
        Entity te = tag_list.tags[i];
        if (te == Entity::none()) {
            continue;
        }

        const FlowTagState &tag = ctx.get<FlowTagState>(te);
        if (tag.flow_id == flow_id) {
            return te;
        }
    }

    return Entity::none();
}

} // namespace

void Sim::pfcPropagateOnePort(Context &ctx,
                              int32_t port_id,
                              PortState &,
                              PortPfcState &pfc_state,
                              DirtyPort &dirty,
                              PortInbox &inbox,
                              PortTraceLast &trace)
{
    (void)ctx;
    (void)port_id;

    bool keep_trace = traceModeEnabled();
    if (keep_trace) {
        trace.pfc_applied = 0;
        trace.pfc_skipped = 0;
    }

    if (enablePfc == 0 || inbox.num_pfc == 0) {
        inbox.num_pfc = 0;
        return;
    }

    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = keep_trace && compiledSystemLogEnabled(scope, step);

    for (int32_t i = 0; i < inbox.num_pfc; i++) {
        const PfcControlEv &ev = inbox.pfcs[i];
        if (ev.priority >= 0 && ev.priority < PFC_MAX_PRIORITY) {
            pfc_state.paused[ev.priority] = ev.paused;
            dirty.isDirty = 1;
            if (keep_trace) {
                trace.pfc_applied += 1;
            }
            if (log_enabled) {
                printSystemPfcState(step, now, ev,
                    pfc_state.paused[ev.priority], dirty.isDirty);
            }
        } else {
            if (keep_trace) {
                trace.pfc_skipped += 1;
            }
        }
    }

    inbox.num_pfc = 0;
}

void Sim::flowArrivalOnePort(Context &ctx,
                             int32_t port_id,
                             PortState &,
                             DirtyPort &dirty,
                             PortInbox &inbox,
                             PortTagList &tag_list,
                             PortCreateList &create_list,
                             PortTraceLast &trace)
{
    bool keep_trace = traceModeEnabled();
    if (keep_trace) {
        trace.arrival_created = 0;
        trace.arrival_updated = 0;
        trace.arrival_skipped = 0;
    }

    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = keep_trace && compiledSystemLogEnabled(scope, step);

    for (int32_t i = 0; i < inbox.num_arrival; i++) {
        const FlowArrivalEv &ev = inbox.arrivals[i];
        if (ev.port_id != port_id) {
            if (keep_trace) {
                trace.arrival_skipped += 1;
            }
            continue;
        }

        Entity existing = findTagInPortList(ctx, tag_list, ev.flow_id);
        if (existing != Entity::none()) {
            FlowTagState &tag = ctx.get<FlowTagState>(existing);
            tag.in_bw = ev.in_bw;
            if (ev.is_source != 0) {
                tag.is_source = 1;
                tag.remaining = ev.size;
            }
            dirty.isDirty = 1;
            if (keep_trace) {
                trace.arrival_updated += 1;
            }
            if (log_enabled) {
                printSystemArrivalTag(step, now, "update", tag, dirty.isDirty);
            }
            continue;
        }

        if (create_list.num < MAX_PORT_CREATE) {
            PortCreateReq &req = create_list.reqs[create_list.num++];
            req.from_arrival = 1;
            req.flow_id = ev.flow_id;
            req.in_bw = ev.in_bw;
            req.size = ev.size;
            req.is_source = ev.is_source;
            req.priority = ev.priority;
            req.log_enabled = log_enabled ? 1 : 0;
            req.log_label = "create";
        } else {
            if (keep_trace) {
                trace.arrival_skipped += 1;
            }
        }
    }

    inbox.num_arrival = 0;
}

void Sim::bwUpdateOnePort(Context &ctx,
                          int32_t port_id,
                          PortState &,
                          PortBuffer &,
                          DirtyPort &dirty,
                          PortInbox &inbox,
                          PortTagList &tag_list,
                          PortCreateList &create_list,
                          PortCleanup &cleanup,
                          PortOutbox &outbox,
                          PortCompletionList &completions,
                          PortTraceLast &trace)
{
    bool keep_trace = traceModeEnabled();
    if (keep_trace) {
        trace.bwupd_created = 0;
        trace.bwupd_updated = 0;
        trace.bwupd_buffered_zero = 0;
        trace.bwupd_destroyed = 0;
        trace.bwupd_forwarded = 0;
        trace.bwupd_completed = 0;
        trace.bwupd_skipped = 0;
    }

    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = keep_trace && compiledSystemLogEnabled(scope, step);

    for (int32_t i = 0; i < inbox.num_bwupd; i++) {
        const BwUpdateEv &ev = inbox.bwupds[i];
        if (ev.port_id != port_id) {
            if (keep_trace) {
                trace.bwupd_skipped += 1;
            }
            continue;
        }

        Entity existing = findTagInPortList(ctx, tag_list, ev.flow_id);

        if (ev.in_bw == 0.0) {
            if (existing != Entity::none()) {
                FlowTagState &tag = ctx.get<FlowTagState>(existing);
                materializeBacklog(tag, now);
                if (enableBuffer != 0 && tag.backlog > 1e-15) {
                    tag.in_bw = 0.0;
                    dirty.isDirty = 1;
                    if (keep_trace) {
                        trace.bwupd_buffered_zero += 1;
                    }
                    if (log_enabled) {
                        printSystemBwUpdateTag(step, now, "buffered_zero",
                            tag, dirty.isDirty);
                    }
                } else {
                    FlowTagState tag_copy = tag;
                    if (cleanup.num < MAX_PORT_CLEANUP) {
                        cleanup.tags[cleanup.num] = existing;
                        cleanup.propagate[cleanup.num] = 1;
                        cleanup.num += 1;
                    }
                    dirty.isDirty = 1;
                    if (keep_trace) {
                        trace.bwupd_destroyed += 1;
                    }
                    if (log_enabled) {
                        printSystemBwUpdateTag(step, now, "destroy",
                            tag_copy, dirty.isDirty);
                    }
                }
            } else {
                if (keep_trace) {
                    trace.bwupd_skipped += 1;
                }
            }
            continue;
        }

        if (existing == Entity::none()) {
            bool has_cleanup = false;
            for (int32_t j = 0; j < inbox.num_bwupd; j++) {
                if (inbox.bwupds[j].flow_id == ev.flow_id &&
                    inbox.bwupds[j].in_bw == 0.0) {
                    has_cleanup = true;
                    break;
                }
            }

            if (has_cleanup) {
                int32_t next_port =
                    lookupFlowRouteNext(ev.flow_id, ev.port_id);
                if (next_port >= 0) {
                    DelayedEvent cleanup_ev {};
                    cleanup_ev.t = computePropagationTimeForPort(
                        ev.port_id, next_port);
                    cleanup_ev.type = DelayedEvent::Type::BwUpdate;
                    cleanup_ev.bwupd = BwUpdateEv {
                        .port_id = next_port,
                        .flow_id = ev.flow_id,
                        .in_bw = 0.0,
                    };
                    if (outbox.num_events < MAX_PORT_OUTBOX) {
                        outbox.events[outbox.num_events++] = cleanup_ev;
                    }
                    if (keep_trace) {
                        trace.bwupd_forwarded += 1;
                    }
                    if (log_enabled) {
                        printSystemBwUpdateForward(step, now, ev.flow_id,
                            ev.port_id, next_port);
                    }
                } else {
                    if (completions.num < MAX_PORT_COMPLETE) {
                        completions.flow_ids[completions.num++] = ev.flow_id;
                    }
                    if (keep_trace) {
                        trace.bwupd_completed += 1;
                    }
                    if (log_enabled) {
                        printSystemBwUpdateComplete(step, now, ev.flow_id);
                    }
                }
                continue;
            }

            if (create_list.num < MAX_PORT_CREATE) {
                int32_t pri = 0;
                const FlowDef *flow_def = getFlowDef(ev.flow_id);
                if (flow_def != nullptr) {
                    pri = flow_def->priority;
                }

                PortCreateReq &req = create_list.reqs[create_list.num++];
                req.from_arrival = 0;
                req.flow_id = ev.flow_id;
                req.in_bw = ev.in_bw;
                req.size = 0.0;
                req.is_source = 0;
                req.priority = pri;
                req.log_enabled = log_enabled ? 1 : 0;
                req.log_label = "create";
            } else {
                if (keep_trace) {
                    trace.bwupd_skipped += 1;
                }
            }
        } else {
            FlowTagState &tag = ctx.get<FlowTagState>(existing);
            if (tag.in_bw != ev.in_bw) {
                materializeBacklog(tag, now);
                tag.in_bw = ev.in_bw;
            }
            dirty.isDirty = 1;
            if (keep_trace) {
                trace.bwupd_updated += 1;
            }
            if (log_enabled) {
                printSystemBwUpdateTag(step, now, "update", tag, dirty.isDirty);
            }
        }
    }

    inbox.num_bwupd = 0;
}

void Sim::flushTagCreate(Context &ctx)
{
    bool keep_trace = traceModeEnabled();
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = keep_trace && compiledSystemLogEnabled(scope, step);

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        PortCreateList &cl = portCreateLists[port_id];
        if (cl.num == 0) {
            continue;
        }

        PortTraceLast &trace = portTraceLasts[port_id];
        for (int32_t i = 0; i < cl.num; i++) {
            const PortCreateReq &req = cl.reqs[i];
            Entity created = createTagOnPort(ctx, port_id, req.flow_id,
                req.in_bw, req.size, req.is_source != 0, req.priority);
            if (created == Entity::none()) {
                if (keep_trace) {
                    if (req.from_arrival != 0) {
                        trace.arrival_skipped += 1;
                    } else {
                        trace.bwupd_skipped += 1;
                    }
                }
                continue;
            }

            if (keep_trace) {
                if (req.from_arrival != 0) {
                    trace.arrival_created += 1;
                } else {
                    trace.bwupd_created += 1;
                }
            }

            if (log_enabled && req.log_enabled != 0) {
                const FlowTagState &tag = ctx.get<FlowTagState>(created);
                int32_t dirty_flag = portDirtyStates[port_id].isDirty;
                if (req.from_arrival != 0) {
                    printSystemArrivalTag(step, now, req.log_label,
                        tag, dirty_flag);
                } else {
                    printSystemBwUpdateTag(step, now, req.log_label,
                        tag, dirty_flag);
                }
            }
        }

        cl.num = 0;
    }
}

void Sim::flushFlowCompletion(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        PortCompletionList &cl = portCompletionLists[port_id];
        for (int32_t i = 0; i < cl.num; i++) {
            recordFlowCompletion(cl.flow_ids[i], now);
        }
        cl.num = 0;
    }
}

void Sim::logIngressChain(Context &ctx)
{
    if (!traceModeEnabled()) {
        return;
    }

    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = compiledSystemLogEnabled(scope, step);

    if (!log_enabled) {
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity port_e = portEntities[port_id];
            if (port_e == Entity::none()) {
                continue;
            }

            PortTraceLast &tr = portTraceLasts[port_id];
            tr.arrival_created = 0;
            tr.arrival_updated = 0;
            tr.arrival_skipped = 0;
            tr.bwupd_created = 0;
            tr.bwupd_updated = 0;
            tr.bwupd_buffered_zero = 0;
            tr.bwupd_destroyed = 0;
            tr.bwupd_forwarded = 0;
            tr.bwupd_completed = 0;
            tr.bwupd_skipped = 0;
            tr.pfc_applied = 0;
            tr.pfc_skipped = 0;
        }
        return;
    }

    int32_t arr_c = 0;
    int32_t arr_u = 0;
    int32_t arr_s = 0;
    int32_t bw_c = 0;
    int32_t bw_u = 0;
    int32_t bw_bz = 0;
    int32_t bw_d = 0;
    int32_t bw_f = 0;
    int32_t bw_cmp = 0;
    int32_t bw_s = 0;
    int32_t pfc_a = 0;
    int32_t pfc_s = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        PortTraceLast &tr = portTraceLasts[port_id];
        arr_c += tr.arrival_created;
        arr_u += tr.arrival_updated;
        arr_s += tr.arrival_skipped;
        bw_c += tr.bwupd_created;
        bw_u += tr.bwupd_updated;
        bw_bz += tr.bwupd_buffered_zero;
        bw_d += tr.bwupd_destroyed;
        bw_f += tr.bwupd_forwarded;
        bw_cmp += tr.bwupd_completed;
        bw_s += tr.bwupd_skipped;
        pfc_a += tr.pfc_applied;
        pfc_s += tr.pfc_skipped;

        tr.arrival_created = 0;
        tr.arrival_updated = 0;
        tr.arrival_skipped = 0;
        tr.bwupd_created = 0;
        tr.bwupd_updated = 0;
        tr.bwupd_buffered_zero = 0;
        tr.bwupd_destroyed = 0;
        tr.bwupd_forwarded = 0;
        tr.bwupd_completed = 0;
        tr.bwupd_skipped = 0;
        tr.pfc_applied = 0;
        tr.pfc_skipped = 0;
    }

    printSystemBegin(step, now, scope, "flow_arrival");
    printSystemArrivalSummary(step, now, arr_c, arr_u, arr_s);
    printSystemEnd(step, now, scope, "flow_arrival");

    printSystemBegin(step, now, scope, "bw_update_ingress");
    printSystemBwUpdateSummary(step, now, bw_c, bw_u, bw_bz, bw_d, bw_f,
        bw_cmp, bw_s);
    printSystemEnd(step, now, scope, "bw_update_ingress");

    if (enablePfc != 0) {
        printSystemBegin(step, now, scope, "pfc_propagate");
        printSystemPfcSummary(step, now, pfc_a, pfc_s);
        printSystemEnd(step, now, scope, "pfc_propagate");
    }
}

}
