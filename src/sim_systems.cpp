#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

int32_t Sim::findSourceTagIndex(FlowId flow_id) const
{
    for (int32_t i = 0; i < numSourceTags; i++) {
        if (sourceTags[i].flow_id == flow_id) {
            return i;
        }
    }
    return -1;
}

int32_t Sim::findIngressTagIndex(int32_t ingress_port_id, FlowId flow_id) const
{
    for (int32_t i = 0; i < numIngressTags; i++) {
        if (ingressTags[i].ingress_port_id == ingress_port_id &&
            ingressTags[i].flow_id == flow_id) {
            return i;
        }
    }
    return -1;
}

Entity Sim::findTag(int32_t port_id, FlowId flow_id) const
{
    for (int32_t i = 0; i < numTagIndexEntries; i++) {
        if (tagIndex[i].port_id == port_id && tagIndex[i].flow_id == flow_id) {
            return tagIndex[i].entity;
        }
    }

    return Entity::none();
}

void Sim::recordFlowCompletion(FlowId flow_id, Time end_time)
{
    for (int32_t i = 0; i < numFlowCompletions; i++) {
        if (flowCompletions[i].flow_id == flow_id) {
            if (end_time > flowCompletions[i].record.end_time) {
                flowCompletions[i].record.end_time = end_time;
            }
            return;
        }
    }

    if (numFlowCompletions >= MAX_FLOW_COMPLETIONS) {
        return;
    }

    for (int32_t i = 0; i < numFlowDefs; i++) {
        if (flowDefs[i].id == flow_id) {
            FlowCompletionEntry &entry = flowCompletions[numFlowCompletions++];
            entry.flow_id = flow_id;
            entry.record = FlowCompletionRecord {
                .flow_id = flow_id,
                .src_node = flowDefs[i].src_node,
                .dst_node = flowDefs[i].dst_node,
                .size = flowDefs[i].size,
                .start_time = flowDefs[i].start_time,
                .end_time = end_time,
                .priority = flowDefs[i].priority,
            };
            return;
        }
    }
}

void Sim::destroyTag(Engine &ctx,
                     Entity tag_entity,
                     bool propagate_cleanup,
                     Time logical_now)
{
    if (tag_entity == Entity::none()) {
        return;
    }

    FlowTagState tag = ctx.get<FlowTagState>(tag_entity);
    Time effective_now = logical_now >= 0.0 ? logical_now : now;

    if (tag.next_port_id < 0) {
        recordFlowCompletion(tag.flow_id, effective_now);
    }

    if (propagate_cleanup && tag.next_port_id >= 0) {
        if (tag.downstream_created != 0) {
            int32_t src_node_slot = findNodeSlot(portToNode[tag.port_id]);
            int32_t dst_node_slot = findNodeSlot(portToNode[tag.next_port_id]);
            Time delay = 0.0;
            if (src_node_slot >= 0 && dst_node_slot >= 0) {
                delay = linkDelays[src_node_slot][dst_node_slot];
                if (delay < 0.0) {
                    delay = 0.0;
                }
            }
            DelayedEvent ev {};
            ev.t = effective_now + delay;
            ev.type = DelayedEvent::Type::BwUpdate;
            ev.bwupd = BwUpdateEv {
                .port_id = tag.next_port_id,
                .flow_id = tag.flow_id,
                .in_bw = 0.0,
            };
            pushDelayedEvent(ev);
        } else {
            int32_t cur = tag.port_id;
            int32_t nxt = lookupFlowRouteNext(tag.flow_id, cur);
            while (nxt >= 0) {
                cur = nxt;
                nxt = lookupFlowRouteNext(tag.flow_id, cur);
            }
            if (cur != tag.port_id) {
                recordFlowCompletion(tag.flow_id, effective_now);
            }
        }
    }

    for (int32_t i = 0; i < numTagIndexEntries; i++) {
        if (tagIndex[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numTagIndexEntries; j++) {
                tagIndex[j - 1] = tagIndex[j];
            }
            numTagIndexEntries -= 1;
            break;
        }
    }

    for (int32_t i = 0; i < numIngressTags; i++) {
        if (ingressTags[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numIngressTags; j++) {
                ingressTags[j - 1] = ingressTags[j];
            }
            numIngressTags -= 1;
            i -= 1;
        }
    }

    for (int32_t i = 0; i < numSourceTags; i++) {
        if (sourceTags[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numSourceTags; j++) {
                sourceTags[j - 1] = sourceTags[j];
            }
            numSourceTags -= 1;
            break;
        }
    }

    if (tag.port_id >= 0 && tag.port_id < numPorts) {
        Entity port_entity = portEntities[tag.port_id];
        if (port_entity != Entity::none()) {
            ctx.get<DirtyPort>(port_entity).isDirty = 1;
        }
    }

    ctx.destroyEntity(tag_entity);
}

Entity Sim::createTagOnPort(Engine &ctx,
                            int32_t port_id,
                            FlowId flow_id,
                            Bw in_bw,
                            Bytes size,
                            bool is_source,
                            int32_t priority)
{
    if (port_id < 0 || port_id >= numPorts) {
        return Entity::none();
    }

    Entity port_entity = portEntities[port_id];
    if (port_entity == Entity::none()) {
        return Entity::none();
    }

    Entity tag_entity = ctx.makeEntity<FlowTag>();
    FlowTagState tag {};
    tag.port_id = port_id;
    tag.flow_id = flow_id;
    tag.priority = priority;
    tag.in_bw = in_bw;
    tag.out_bw = 0.0;
    tag.prev_out_bw = 0.0;
    tag.backlog = 0.0;
    tag.last_backlog_time = now;
    tag.remaining = size;
    tag.last_remaining_time = now;
    tag.is_source = is_source ? 1 : 0;
    tag.downstream_created = 0;
    tag.next_port_id = lookupFlowRouteNext(flow_id, port_id);
    tag.ingress_port_id = -1;

    if (!is_source) {
        for (int32_t i = 0; i < numFlowRoutes; i++) {
            if (flowRoutes[i].flow_id != flow_id) {
                continue;
            }
            for (int32_t j = 0; j < flowRoutes[i].num_steps; j++) {
                if (flowRoutes[i].steps[j].next_port_id == port_id) {
                    int32_t upstream_port = flowRoutes[i].steps[j].port_id;
                    if (upstream_port >= 0 && upstream_port < numPorts) {
                        tag.ingress_port_id = peerPort[upstream_port];
                    }
                    break;
                }
            }
        }
    }

    ctx.get<FlowTagState>(tag_entity) = tag;

    if (numTagIndexEntries < MAX_TAG_INDEX) {
        tagIndex[numTagIndexEntries++] = TagIndexEntry {
            .port_id = port_id,
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    if (tag.ingress_port_id >= 0 && numIngressTags < MAX_INGRESS_TAGS) {
        ingressTags[numIngressTags++] = IngressTagEntry {
            .ingress_port_id = tag.ingress_port_id,
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    if (is_source && numSourceTags < MAX_SOURCE_TAGS) {
        sourceTags[numSourceTags++] = SourceTagEntry {
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    ctx.get<DirtyPort>(port_entity).isDirty = 1;
    PortBuffer &port_buf = ctx.get<PortBuffer>(port_entity);
    if (port_buf.last_update_time < now) {
        port_buf.last_update_time = now;
    }

    return tag_entity;
}

void Sim::injectFlow(int32_t src_port_id, const FlowDef &flow)
{
    int32_t src_slot = findNodeSlot(flow.src_node);
    Bw src_in_bw = 0.0;
    if (src_slot >= 0) {
        src_in_bw = topoNodes[src_slot].port_bw;
    }

    DelayedEvent ev {};
    ev.t = now;
    ev.type = DelayedEvent::Type::Arrival;
    ev.arrival = FlowArrivalEv {
        .port_id = src_port_id,
        .flow_id = flow.id,
        .size = flow.size,
        .in_bw = src_in_bw,
        .is_source = 1,
        .priority = flow.priority,
    };
    pushDelayedEvent(ev);
}

void Sim::injectFlowDef(const FlowDef &flow)
{
    NodeId path[MAX_PATH_NODES] {};
    int32_t path_len = getPath(flow.src_node, flow.dst_node, flow.id,
        path, MAX_PATH_NODES);
    if (path_len < 2) {
        return;
    }

    int32_t src_slot = findNodeSlot(flow.src_node);
    if (src_slot < 0) {
        return;
    }

    NodeId first_hop = path[1];
    int32_t first_neighbor_idx = findNeighborSlot(src_slot, first_hop);
    if (first_neighbor_idx < 0) {
        return;
    }

    int32_t port_path[MAX_FLOW_ROUTE_STEPS + 1] {};
    int32_t port_path_len = 0;

    for (int32_t i = 0; i + 1 < path_len; i++) {
        int32_t node_slot = findNodeSlot(path[i]);
        if (node_slot < 0) {
            return;
        }
        int32_t neighbor_idx = findNeighborSlot(node_slot, path[i + 1]);
        if (neighbor_idx < 0) {
            return;
        }
        port_path[port_path_len++] = topoNodes[node_slot].neighbors[neighbor_idx].port_id;
    }
    port_path[port_path_len++] = -1;

    bool exists = false;
    for (int32_t i = 0; i < numFlowRoutes; i++) {
        if (flowRoutes[i].flow_id == flow.id) {
            exists = true;
            break;
        }
    }

    if (!exists && numFlowRoutes < MAX_FLOWS) {
        FlowRouteState &route = flowRoutes[numFlowRoutes++];
        route.flow_id = flow.id;
        route.num_steps = port_path_len - 1;
        for (int32_t i = 0; i + 1 < port_path_len; i++) {
            route.steps[i].port_id = port_path[i];
            route.steps[i].next_port_id = port_path[i + 1];
        }
    }

    int32_t src_port_id = topoNodes[src_slot].neighbors[first_neighbor_idx].port_id;
    injectFlow(src_port_id, flow);
}

void Sim::schedulePendingFlows()
{
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    int32_t pending_before = numPendingFlows;
    int32_t delayed_before = numDelayedEvents;
    int32_t flow_routes_before = numFlowRoutes;
    int32_t scheduled_count = 0;
    FlowDef logged_flows[MAX_FLOWS] {};
    int32_t num_logged_flows = 0;

    if (log_enabled) {
        printSystemBegin(step, now, scope, "schedule_pending_flows");
    }

    while (numPendingFlows > 0 && pendingFlows[0].start_time <= now + 1e-15) {
        FlowDef flow = pendingFlows[0];
        for (int32_t i = 1; i < numPendingFlows; i++) {
            pendingFlows[i - 1] = pendingFlows[i];
        }
        numPendingFlows -= 1;
        if (log_enabled && num_logged_flows < MAX_FLOWS) {
            logged_flows[num_logged_flows++] = flow;
        }
        scheduled_count += 1;
        injectFlowDef(flow);
    }

    if (log_enabled) {
        for (int32_t i = 0; i < num_logged_flows; i++) {
            for (int32_t j = i + 1; j < num_logged_flows; j++) {
                if (logged_flows[j].id < logged_flows[i].id) {
                    FlowDef tmp = logged_flows[i];
                    logged_flows[i] = logged_flows[j];
                    logged_flows[j] = tmp;
                }
            }
        }
        for (int32_t i = 0; i < num_logged_flows; i++) {
            printSystemScheduleFlow(step, now, logged_flows[i]);
        }
        printSystemScheduleSummary(step, now,
            scheduled_count,
            pending_before,
            numPendingFlows,
            delayed_before,
            numDelayedEvents,
            flow_routes_before,
            numFlowRoutes);
        printSystemEnd(step, now, scope, "schedule_pending_flows");
    }
}

void Sim::deliverEvents()
{
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    int32_t delayed_before = numDelayedEvents;

    if (log_enabled) {
        printSystemBegin(step, now, scope, "deliver_events");
    }

    numInboxArrival = 0;
    numInboxBwUpdate = 0;
    numInboxPfc = 0;

    int32_t write_idx = 0;
    for (int32_t i = 0; i < numDelayedEvents; i++) {
        if (delayedEvents[i].t <= now + 1e-15) {
            if (delayedEvents[i].type == DelayedEvent::Type::Arrival) {
                if (numInboxArrival < MAX_EVENTS_PER_STEP) {
                    inboxArrival[numInboxArrival++] = delayedEvents[i].arrival;
                    if (log_enabled) {
                        printSystemDeliverArrival(step, now, delayedEvents[i].arrival);
                    }
                }
            } else if (delayedEvents[i].type == DelayedEvent::Type::BwUpdate) {
                if (numInboxBwUpdate < MAX_EVENTS_PER_STEP) {
                    inboxBwUpdate[numInboxBwUpdate++] = delayedEvents[i].bwupd;
                    if (log_enabled) {
                        printSystemDeliverBwUpdate(step, now, delayedEvents[i].bwupd);
                    }
                }
            } else {
                if (numInboxPfc < MAX_EVENTS_PER_STEP) {
                    inboxPfc[numInboxPfc++] = delayedEvents[i].pfcctrl;
                    if (log_enabled) {
                        printSystemDeliverPfc(step, now, delayedEvents[i].pfcctrl);
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

void Sim::flowArrivalSystem(Engine &ctx)
{
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    int32_t created_count = 0;
    int32_t updated_count = 0;
    int32_t skipped_count = 0;

    if (log_enabled) {
        printSystemBegin(step, now, scope, "flow_arrival");
    }

    for (int32_t i = 0; i < numInboxArrival; i++) {
        const FlowArrivalEv &ev = inboxArrival[i];
        if (ev.port_id < 0 || ev.port_id >= numPorts) {
            skipped_count += 1;
            continue;
        }

        Entity port_entity = portEntities[ev.port_id];
        if (port_entity == Entity::none()) {
            skipped_count += 1;
            continue;
        }

        Entity existing = findTag(ev.port_id, ev.flow_id);
        if (existing != Entity::none()) {
            FlowTagState &tag = ctx.get<FlowTagState>(existing);
            tag.in_bw = ev.in_bw;
            if (ev.is_source != 0) {
                tag.is_source = 1;
                tag.remaining = ev.size;
            }
            ctx.get<DirtyPort>(port_entity).isDirty = 1;
            updated_count += 1;
            if (log_enabled) {
                printSystemArrivalTag(step, now, "update", tag,
                    ctx.get<DirtyPort>(port_entity).isDirty);
            }
            continue;
        }

        Entity created = createTagOnPort(ctx, ev.port_id, ev.flow_id, ev.in_bw,
            ev.size, ev.is_source != 0, ev.priority);
        if (created == Entity::none()) {
            skipped_count += 1;
            continue;
        }
        created_count += 1;
        if (log_enabled) {
            printSystemArrivalTag(step, now, "create",
                ctx.get<FlowTagState>(created),
                ctx.get<DirtyPort>(port_entity).isDirty);
        }
    }

    if (log_enabled) {
        printSystemArrivalSummary(step, now,
            created_count,
            updated_count,
            skipped_count);
        printSystemEnd(step, now, scope, "flow_arrival");
    }
}

void Sim::bwUpdateIngressSystem(Engine &ctx)
{
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    int32_t created_count = 0;
    int32_t updated_count = 0;
    int32_t buffered_zero_count = 0;
    int32_t destroyed_count = 0;
    int32_t forwarded_count = 0;
    int32_t completed_count = 0;
    int32_t skipped_count = 0;

    if (log_enabled) {
        printSystemBegin(step, now, scope, "bw_update_ingress");
    }

    for (int32_t i = 0; i < numInboxBwUpdate; i++) {
        const BwUpdateEv &ev = inboxBwUpdate[i];
        if (ev.port_id < 0 || ev.port_id >= numPorts) {
            skipped_count += 1;
            continue;
        }

        Entity port_entity = portEntities[ev.port_id];
        if (port_entity == Entity::none()) {
            skipped_count += 1;
            continue;
        }

        Entity existing = findTag(ev.port_id, ev.flow_id);

        if (ev.in_bw == 0.0) {
            if (existing != Entity::none()) {
                FlowTagState &tag = ctx.get<FlowTagState>(existing);
                materializeBacklog(tag, now);
                if (enableBuffer != 0 && tag.backlog > 1e-15) {
                    tag.in_bw = 0.0;
                    ctx.get<DirtyPort>(port_entity).isDirty = 1;
                    buffered_zero_count += 1;
                    if (log_enabled) {
                        printSystemBwUpdateTag(step, now, "buffered_zero", tag,
                            ctx.get<DirtyPort>(port_entity).isDirty);
                    }
                } else {
                    FlowTagState tag_copy = tag;
                    destroyTag(ctx, existing, true, now);
                    destroyed_count += 1;
                    if (log_enabled) {
                        printSystemBwUpdateTag(step, now, "destroy", tag_copy,
                            ctx.get<DirtyPort>(port_entity).isDirty);
                    }
                }
            } else {
                skipped_count += 1;
            }
            continue;
        }

        if (existing == Entity::none()) {
            bool has_cleanup = false;
            for (int32_t j = 0; j < numInboxBwUpdate; j++) {
                if (inboxBwUpdate[j].port_id == ev.port_id &&
                    inboxBwUpdate[j].flow_id == ev.flow_id &&
                    inboxBwUpdate[j].in_bw == 0.0) {
                    has_cleanup = true;
                    break;
                }
            }

            if (has_cleanup) {
                int32_t next_port = lookupFlowRouteNext(ev.flow_id, ev.port_id);
                if (next_port >= 0) {
                    DelayedEvent cleanup_ev {};
                    cleanup_ev.t = computePropagationTimeForPort(ev.port_id, next_port);
                    cleanup_ev.type = DelayedEvent::Type::BwUpdate;
                    cleanup_ev.bwupd = BwUpdateEv {
                        .port_id = next_port,
                        .flow_id = ev.flow_id,
                        .in_bw = 0.0,
                    };
                    pushDelayedEvent(cleanup_ev);
                    forwarded_count += 1;
                    if (log_enabled) {
                        printSystemBwUpdateForward(step, now, ev.flow_id, ev.port_id, next_port);
                    }
                } else {
                    recordFlowCompletion(ev.flow_id, now);
                    completed_count += 1;
                    if (log_enabled) {
                        printSystemBwUpdateComplete(step, now, ev.flow_id);
                    }
                }
                continue;
            }

            int32_t pri = 0;
            for (int32_t j = 0; j < numFlowDefs; j++) {
                if (flowDefs[j].id == ev.flow_id) {
                    pri = flowDefs[j].priority;
                    break;
                }
            }
            Entity created = createTagOnPort(ctx, ev.port_id, ev.flow_id, ev.in_bw, 0.0, false, pri);
            if (created == Entity::none()) {
                skipped_count += 1;
                continue;
            }
            created_count += 1;
            if (log_enabled) {
                printSystemBwUpdateTag(step, now, "create", ctx.get<FlowTagState>(created),
                    ctx.get<DirtyPort>(port_entity).isDirty);
            }
        } else {
            FlowTagState &tag = ctx.get<FlowTagState>(existing);
            if (tag.in_bw != ev.in_bw) {
                materializeBacklog(tag, now);
                tag.in_bw = ev.in_bw;
            }
            ctx.get<DirtyPort>(port_entity).isDirty = 1;
            updated_count += 1;
            if (log_enabled) {
                printSystemBwUpdateTag(step, now, "update", tag,
                    ctx.get<DirtyPort>(port_entity).isDirty);
            }
        }
    }

    if (log_enabled) {
        printSystemBwUpdateSummary(step, now,
            created_count,
            updated_count,
            buffered_zero_count,
            destroyed_count,
            forwarded_count,
            completed_count,
            skipped_count);
        printSystemEnd(step, now, scope, "bw_update_ingress");
    }
}

void Sim::clearDirtyPorts(Engine &ctx)
{
    constexpr const char *scope = "emit_pfc";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    numLastDirtyPortIDs = 0;
    int32_t cleared_port_count = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        DirtyPort &dirty = ctx.get<DirtyPort>(port_e);
        if (dirty.isDirty != 0) {
            if (numLastDirtyPortIDs < MAX_TOPO_PORTS) {
                lastDirtyPortIDs[numLastDirtyPortIDs++] = port_id;
            }
            dirty.isDirty = 0;
            cleared_port_count += 1;
        }
    }

    if (log_enabled) {
        printSystemClearSummary(step, now, cleared_port_count);
    }
}

void Sim::flowProgressAndCleanupSystem(Engine &ctx, Time dt)
{
    Time next_now = now + dt;
    bool need_check_finish = cachedNextFinishTime < std::numeric_limits<Time>::max() && cachedNextFinishTime <= dt + 1e-12;

    if (need_check_finish) {
        Entity finished[MAX_SOURCE_TAGS] {};
        int32_t num_finished = 0;
        Time next_finish = std::numeric_limits<Time>::max();
        for (int32_t i = 0; i < numSourceTags; i++) {
            Entity tag_e = sourceTags[i].entity;
            if (tag_e == Entity::none()) {
                continue;
            }
            FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            materializeRemaining(tag, next_now);
            if (tag.remaining < 1.0) {
                tag.remaining = 0.0;
                if (num_finished < MAX_SOURCE_TAGS) {
                    finished[num_finished++] = tag_e;
                }
                if (tag.next_port_id >= 0) {
                    DelayedEvent ev {};
                    ev.t = next_now + computePropagationTimeForPort(tag.port_id, tag.next_port_id) - now;
                    ev.type = DelayedEvent::Type::BwUpdate;
                    ev.bwupd = BwUpdateEv {
                        .port_id = tag.next_port_id,
                        .flow_id = tag.flow_id,
                        .in_bw = 0.0,
                    };
                    pushDelayedEvent(ev);
                }
            } else if (tag.out_bw > 1e-15) {
                Time t_finish = tag.remaining / tag.out_bw;
                if (t_finish > 1e-15) {
                    next_finish = std::min(next_finish, t_finish);
                }
            }
        }
        for (int32_t i = 0; i < num_finished; i++) {
            destroyTag(ctx, finished[i], false, next_now);
        }
        cachedNextFinishTime = next_finish;
    } else if (cachedNextFinishTime < std::numeric_limits<Time>::max()) {
        cachedNextFinishTime -= dt;
        if (cachedNextFinishTime < 1e-15) {
            cachedNextFinishTime = 1e-15;
        }
    }

    if (cachedNextDrainTime < std::numeric_limits<Time>::max()) {
        cachedNextDrainTime -= dt;
        if (cachedNextDrainTime < 1e-15) {
            cachedNextDrainTime = std::numeric_limits<Time>::max();
            cachedDrainPortID = -1;
        }
    }

    int32_t i = 0;
    while (i < numBacklogDrainTimers) {
        backlogDrainTimers[i] -= dt;
        if (backlogDrainTimers[i] < 1e-15) {
            int32_t pid = backlogDrainPortIDs[i];
            if (pid >= 0 && pid < numPorts) {
                Entity pe = portEntities[pid];
                if (pe != Entity::none()) {
                    ctx.get<DirtyPort>(pe).isDirty = 1;
                }
            }
            clearBacklogDrainTimer(backlogDrainPortIDs[i]);
        } else {
            i += 1;
        }
    }

    auto mark_egress_ports_for_ingress = [&](int32_t ingress_port) {
        for (int32_t j = 0; j < numIngressTags; j++) {
            if (ingressTags[j].ingress_port_id != ingress_port) {
                continue;
            }
            Entity te = ingressTags[j].entity;
            if (te == Entity::none()) {
                continue;
            }
            FlowTagState &t = ctx.get<FlowTagState>(te);
            if (t.port_id >= 0 && t.port_id < numPorts) {
                Entity pe = portEntities[t.port_id];
                if (pe != Entity::none()) {
                    ctx.get<DirtyPort>(pe).isDirty = 1;
                }
            }
        }
    };

    i = 0;
    while (i < numPfcPauseTimers) {
        pfcPauseTimers[i] -= dt;
        if (pfcPauseTimers[i] < 1e-9) {
            int32_t ingress_port = pfcPausePortIDs[i];
            mark_egress_ports_for_ingress(ingress_port);
            clearPfcPauseTimer(ingress_port);
        } else {
            i += 1;
        }
    }

    i = 0;
    while (i < numPfcResumeTimers) {
        pfcResumeTimers[i] -= dt;
        if (pfcResumeTimers[i] < 1e-9) {
            int32_t ingress_port = pfcResumePortIDs[i];
            mark_egress_ports_for_ingress(ingress_port);
            clearPfcResumeTimer(ingress_port);
        } else {
            i += 1;
        }
    }

    bool all_exhausted = cachedNextDrainTime >= std::numeric_limits<Time>::max() &&
                         cachedNextFinishTime >= std::numeric_limits<Time>::max() &&
                         numDelayedEvents == 0 && numPendingFlows == 0 &&
                         numBacklogDrainTimers == 0 && numPfcPauseTimers == 0 &&
                         numPfcResumeTimers == 0;

    if (all_exhausted) {
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity pe = portEntities[port_id];
            if (pe == Entity::none()) {
                continue;
            }
            PortBuffer &pb_all = ctx.get<PortBuffer>(pe);
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                PriorityBuffer &pb = pb_all.prior_bufs[pri];
                if (pb.buf_cnt > 1e-15 && pb.num_chunks > 0) {
                    ctx.get<DirtyPort>(pe).isDirty = 1;
                    break;
                }
            }
        }

        if (enablePfc != 0) {
            for (int32_t ingress_port = 0; ingress_port < numPorts; ingress_port++) {
                Entity ingress_e = portEntities[ingress_port];
                if (ingress_e == Entity::none()) {
                    continue;
                }
                PortPfcState &pfc = ctx.get<PortPfcState>(ingress_e);
                bool has_active_pause = false;
                for (int32_t p = 0; p < PFC_MAX_PRIORITY; p++) {
                    if (pfc.pause_active[p] != 0) {
                        has_active_pause = true;
                        break;
                    }
                }
                if (!has_active_pause) {
                    continue;
                }

                bool no_tags = true;
                for (int32_t j = 0; j < numIngressTags; j++) {
                    if (ingressTags[j].ingress_port_id == ingress_port && ingressTags[j].entity != Entity::none()) {
                        no_tags = false;
                        break;
                    }
                }

                if (no_tags) {
                    for (int32_t p = 0; p < PFC_MAX_PRIORITY; p++) {
                        if (pfc.pause_active[p] == 0) {
                            continue;
                        }
                        pfc.pause_active[p] = 0;
                        for (int32_t k = 0; k < pfc.paused_upstream_count[p]; k++) {
                            int32_t up = pfc.paused_upstreams[p][k];
                            if (up < 0 || up >= numPorts) {
                                continue;
                            }
                            int32_t detect_slot = findNodeSlot(portToNode[ingress_port]);
                            int32_t upstream_slot = findNodeSlot(portToNode[up]);
                            Time pfc_delay = defaultLinkDelay;
                            if (detect_slot >= 0 && upstream_slot >= 0 && linkDelays[detect_slot][upstream_slot] >= 0.0) {
                                pfc_delay = linkDelays[detect_slot][upstream_slot];
                            }
                            DelayedEvent ev {};
                            ev.t = now + dt + pfc_delay;
                            ev.type = DelayedEvent::Type::PfcControl;
                            ev.pfcctrl = PfcControlEv {
                                .target_port_id = up,
                                .source_port_id = ingress_port,
                                .priority = p,
                                .paused = 0,
                            };
                            pushDelayedEvent(ev);
                        }
                        pfc.paused_upstream_count[p] = 0;
                    }
                    continue;
                }

                double port_buf_cnt[MAX_TOPO_PORTS][PFC_MAX_PRIORITY] {};
                bool any_capped = false;
                for (int32_t j = 0; j < numIngressTags; j++) {
                    if (ingressTags[j].ingress_port_id != ingress_port) {
                        continue;
                    }
                    Entity te = ingressTags[j].entity;
                    if (te == Entity::none()) {
                        continue;
                    }
                    FlowTagState &t = ctx.get<FlowTagState>(te);
                    Entity pe = portEntities[t.port_id];
                    if (pe != Entity::none()) {
                        PortBuffer &pb = ctx.get<PortBuffer>(pe);
                        materializeBufCnt(pb, now);
                        int32_t pri = std::clamp(t.priority, 0, PFC_MAX_PRIORITY - 1);
                        port_buf_cnt[t.port_id][pri] = pb.prior_bufs[pri].buf_cnt;
                    }
                }

                for (int32_t j = 0; j < numIngressTags; j++) {
                    if (ingressTags[j].ingress_port_id != ingress_port) {
                        continue;
                    }
                    Entity te = ingressTags[j].entity;
                    if (te == Entity::none()) {
                        continue;
                    }
                    FlowTagState &t = ctx.get<FlowTagState>(te);
                    if (t.is_source != 0) {
                        continue;
                    }
                    materializeBacklog(t, now);
                    int32_t pri = std::clamp(t.priority, 0, PFC_MAX_PRIORITY - 1);
                    double actual = port_buf_cnt[t.port_id][pri];
                    if (actual < 1.0 && t.backlog > 1.0) {
                        t.backlog = actual;
                        t.last_backlog_time = now;
                        any_capped = true;
                    }
                }

                if (any_capped) {
                    for (int32_t j = 0; j < numIngressTags; j++) {
                        if (ingressTags[j].ingress_port_id != ingress_port) {
                            continue;
                        }
                        Entity te = ingressTags[j].entity;
                        if (te == Entity::none()) {
                            continue;
                        }
                        FlowTagState &t = ctx.get<FlowTagState>(te);
                        if (t.port_id >= 0 && t.port_id < numPorts) {
                            Entity pe = portEntities[t.port_id];
                            if (pe != Entity::none()) {
                                ctx.get<DirtyPort>(pe).isDirty = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

}
