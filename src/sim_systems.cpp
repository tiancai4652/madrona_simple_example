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

void Sim::removeFlowDef(FlowId flow_id)
{
    for (int32_t i = 0; i < numFlowDefs; i++) {
        if (flowDefs[i].id != flow_id) {
            continue;
        }
        for (int32_t j = i + 1; j < numFlowDefs; j++) {
            flowDefs[j - 1] = flowDefs[j];
        }
        numFlowDefs -= 1;
        flowDefs[numFlowDefs] = FlowDef {};
        return;
    }
}

void Sim::removeFlowRoute(FlowId flow_id)
{
    for (int32_t i = 0; i < numFlowRoutes; i++) {
        if (flowRoutes[i].flow_id != flow_id) {
            continue;
        }
        for (int32_t j = i + 1; j < numFlowRoutes; j++) {
            flowRoutes[j - 1] = flowRoutes[j];
        }
        numFlowRoutes -= 1;
        flowRoutes[numFlowRoutes] = FlowRouteState {};
        return;
    }
}

void Sim::recordFlowCompletion(FlowId flow_id, Time end_time)
{
    for (int32_t i = 0; i < numFlowCompletions; i++) {
        if (flowCompletions[i].flow_id == flow_id) {
            if (end_time > flowCompletions[i].record.end_time) {
                flowCompletions[i].record.end_time = end_time;
            }
            removeFlowRoute(flow_id);
            removeFlowDef(flow_id);
            return;
        }
    }

    if (numFlowCompletions >= MAX_FLOW_COMPLETIONS) {
        removeFlowRoute(flow_id);
        removeFlowDef(flow_id);
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
            removeFlowRoute(flow_id);
            removeFlowDef(flow_id);
            return;
        }
    }

    removeFlowRoute(flow_id);
}

void Sim::destroyTag(Context &ctx,
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

    // Phase D: mirror the removal in the owning port's PortTagList using
    // swap-last so the per-Port workers no longer need the global tagIndex
    // scan. FlowTagState.port_entity is the cached owning port.
    if (tag.port_entity != Entity::none()) {
        PortTagList &ptl = ctx.get<PortTagList>(tag.port_entity);
        for (int32_t i = 0; i < ptl.count; i++) {
            if (ptl.tags[i] == tag_entity) {
                ptl.tags[i] = ptl.tags[ptl.count - 1];
                ptl.tags[ptl.count - 1] = Entity::none();
                ptl.count -= 1;
                break;
            }
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

Entity Sim::createTagOnPort(Context &ctx,
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

    // Phase D: cache the owning Port entity so destroyTag can O(1)-lookup
    // PortTagList without rescanning portEntities[].
    tag.port_entity = port_entity;

    ctx.get<FlowTagState>(tag_entity) = tag;

    if (numTagIndexEntries < MAX_TAG_INDEX) {
        tagIndex[numTagIndexEntries++] = TagIndexEntry {
            .port_id = port_id,
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    // Phase D: mirror the new tag entity into the owning port's PortTagList
    // so per-Port workers can iterate only this port's tags. Overflow here
    // would silently drop the tag from the fast list, but tagIndex stays
    // authoritative; the smoke test surfaces MAX_TAGS_PER_PORT overflow
    // before parity runs.
    {
        PortTagList &ptl = ctx.get<PortTagList>(port_entity);
        if (ptl.count < MAX_TAGS_PER_PORT) {
            ptl.tags[ptl.count++] = tag_entity;
        }
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
    Bw src_in_bw = 0.0;
    if (src_port_id >= 0 && src_port_id < numPorts) {
        int32_t src_node = portToNode[src_port_id];
        int32_t src_slot = findNodeSlot(src_node);
        if (src_slot >= 0) {
            const TopoNodeState &node = topoNodes[src_slot];
            for (int32_t i = 0; i < node.num_neighbors; i++) {
                if (node.neighbors[i].port_id == src_port_id) {
                    src_in_bw = topoLinks[src_port_id].bandwidth > 0.0
                        ? topoLinks[src_port_id].bandwidth
                        : node.port_bw;
                    break;
                }
            }
        }
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

void Sim::deliverEvents(Context &ctx)
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

    // Phase E: also reset every Port's PortInbox / PortCreateList /
    // PortCompletionList before dispatching this frame's due events.
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortInbox &inbox = ctx.get<PortInbox>(port_e);
        inbox.num_arrival = 0;
        inbox.num_bwupd = 0;
        inbox.num_pfc = 0;
        ctx.get<PortCreateList>(port_e).num = 0;
        ctx.get<PortCompletionList>(port_e).num = 0;
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
                        PortInbox &inbox = ctx.get<PortInbox>(port_e);
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
                        PortInbox &inbox = ctx.get<PortInbox>(port_e);
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
                        PortInbox &inbox = ctx.get<PortInbox>(port_e);
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

// Phase B.1 singleton: consume the was_dirty_at_clear snapshots that the
// per-Port clearDirtyOnePortSystem produced in parallel, and rebuild
// lastDirtyPortIDs in port_id ascending order (same order the previous
// sequential clearDirtyPorts used). Logging stays bit-for-bit identical.
void Sim::snapshotDirtyPorts(Context &ctx)
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
        PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
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

// Phase B.2 singleton: consume PortCachedHints produced in parallel by
// allocOnePort and fold them into the global cachedNextDrainTime /
// cachedDrainPortID / cachedNextFinishTime. Walks portEntities[] in
// port_id ascending order so the chosen drain/finish port is identical to
// the legacy sequential loop. Also reimplements the legacy rule that if
// the currently cached drain port is being re-processed this frame, its
// global drain cache is reset before taking new per-port hints.
void Sim::reducePortCachedHints(Context &ctx)
{
    bool reset_drain = false;
    if (cachedDrainPortID >= 0 && cachedDrainPortID < numPorts) {
        Entity cached_port_e = portEntities[cachedDrainPortID];
        if (cached_port_e != Entity::none()) {
            if (ctx.get<PortTraceLast>(cached_port_e).was_dirty_at_alloc != 0) {
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
        PortCachedHints &hints = ctx.get<PortCachedHints>(port_e);
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

// Phase B.2 singleton: walk ports in ascending id order and materialize the
// PortDrainHint write requests into backlogDrainTimers. Clears happen first,
// then sets with the usual min-take-if-shorter semantics. Each port only
// touches its own entry, but sequencing via a singleton keeps ordering
// identical across backends.
void Sim::flushPortDrainHints(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortDrainHint &hint = ctx.get<PortDrainHint>(port_e);
        if (hint.want_clear != 0) {
            clearBacklogDrainTimer(port_id);
        }
    }
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortDrainHint &hint = ctx.get<PortDrainHint>(port_e);
        if (hint.want_set != 0) {
            int32_t idx = findBacklogDrainTimerIndex(port_id);
            if (idx < 0 || hint.set_t < backlogDrainTimers[idx]) {
                setBacklogDrainTimer(port_id, hint.set_t);
            }
        }
    }
}

// Phase B.2 singleton: replay the destroyTag calls that allocOnePort
// deferred into PortCleanup. Walks ports in ascending id so tagIndex /
// sourceTags compaction is identical across CPU/GPU.
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
            destroyTag(ctx, cleanup.tags[i], cleanup.propagate[i] != 0, now);
        }
        cleanup.num = 0;
    }
}

// Phase B.2 singleton: re-emit the "alloc" scope log lines using the
// PortTraceLast snapshot that allocOnePort recorded in parallel. We walk
// ports in port_id ascending order so the log output matches the legacy
// sequential implementation byte-for-byte on the happy path.
void Sim::logAllocTraces(Context &ctx)
{
    constexpr const char *scope = "alloc";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);

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

    if (log_enabled) {
        printSystemAllocSummary(step, now, num_dirty, processed, dirty_tag_count);
    }
}

// Phase C singleton: flush per-Port PortOutbox into Sim::delayedEvents in
// port_id ascending order (within a port, keep insertion order). Runs once
// after pfcDetectOnePort (to flush PFC events) and again after emitOnePort
// (to flush Arrival/BwUpdate events), matching the legacy sequential
// "all ports pfc then all ports emit" push order.
void Sim::flushPortOutbox(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortOutbox &outbox = ctx.get<PortOutbox>(port_e);
        for (int32_t i = 0; i < outbox.num_events; i++) {
            pushDelayedEvent(outbox.events[i]);
        }
        outbox.num_events = 0;
    }
}

// Phase C singleton: apply per-Port deferred PFC timer want_* mutations in
// port_id ascending order. Mirrors the legacy order "clear-then-set" per
// port (egress mode's state_changed case clears, ingress mode may set
// pause/resume afterwards within the same port).
void Sim::flushPortPfcTimers(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortPfcState &state = ctx.get<PortPfcState>(port_e);
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

// Phase C singleton: aggregate per-Port PortTraceLast pfc_detect_* fields
// into the emit_pfc scope's printSystemPfcDetectSummary line. Runs after
// flushPortPfcTimers so the printed pfc timer counts are final.
void Sim::logPfcDetectTraces(Context &ctx)
{
    constexpr const char *scope = "emit_pfc";
    uint64_t step = systemLogStep;
    if (!systemLogEnabled(scope, step)) {
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
        numPfcPauseTimers, numPfcResumeTimers);
}

// Phase C singleton: aggregate per-Port PortTraceLast emit_* fields into
// the emit_pfc scope's printSystemEmitSummary line. Runs after the second
// flushPortOutbox so per-port emit state is consistent for the log.
void Sim::logEmitTraces(Context &ctx)
{
    constexpr const char *scope = "emit_pfc";
    uint64_t step = systemLogStep;
    if (!systemLogEnabled(scope, step)) {
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

void Sim::flowProgressAndCleanupSystem(Context &ctx, Time dt)
{
    constexpr const char *scope = "progress";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    Time next_now = now + dt;
    bool need_check_finish = cachedNextFinishTime < std::numeric_limits<Time>::max() && cachedNextFinishTime <= dt + 1e-12;
    int32_t finished_source_count = 0;
    int32_t emitted_cleanup_count = 0;

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
                finished_source_count += 1;
                if (tag.next_port_id >= 0) {
                    Time link_delay = defaultLinkDelay;
                    int32_t src_node_slot = findNodeSlot(portToNode[tag.port_id]);
                    int32_t dst_node_slot = findNodeSlot(portToNode[tag.next_port_id]);
                    if (src_node_slot >= 0 && dst_node_slot >= 0 &&
                        linkDelays[src_node_slot][dst_node_slot] >= 0.0) {
                        link_delay = linkDelays[src_node_slot][dst_node_slot];
                    }

                    DelayedEvent ev {};
                    ev.t = next_now + link_delay;
                    ev.type = DelayedEvent::Type::BwUpdate;
                    ev.bwupd = BwUpdateEv {
                        .port_id = tag.next_port_id,
                        .flow_id = tag.flow_id,
                        .in_bw = 0.0,
                    };
                    pushDelayedEvent(ev);
                    emitted_cleanup_count += 1;
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

    if (log_enabled) {
        double next_finish_gap = cachedNextFinishTime < std::numeric_limits<Time>::max()
            ? cachedNextFinishTime : std::numeric_limits<double>::max();
        printSystemProgressSummary(step, now, dt, finished_source_count,
            emitted_cleanup_count, next_now, next_finish_gap);
    }
}

// ============================================================================
// Phase E: per-Port ingress-chain workers and flush/log singletons.
// ============================================================================

// Helper: find a tag belonging to this port by flow_id, using the
// PortTagList mirror. O(tag_list.count) instead of O(numTagIndexEntries).
static Entity findTagInPortList(Context &ctx, const PortTagList &tag_list, FlowId flow_id)
{
    for (int32_t i = 0; i < tag_list.count; i++) {
        Entity te = tag_list.tags[i];
        if (te == Entity::none()) {
            continue;
        }
        const FlowTagState &t = ctx.get<FlowTagState>(te);
        if (t.flow_id == flow_id) {
            return te;
        }
    }
    return Entity::none();
}

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
    trace.pfc_applied = 0;
    trace.pfc_skipped = 0;
    if (enablePfc == 0 || inbox.num_pfc == 0) {
        inbox.num_pfc = 0;
        return;
    }

    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);

    for (int32_t i = 0; i < inbox.num_pfc; i++) {
        const PfcControlEv &ev = inbox.pfcs[i];
        if (ev.priority >= 0 && ev.priority < PFC_MAX_PRIORITY) {
            pfc_state.paused[ev.priority] = ev.paused;
            dirty.isDirty = 1;
            trace.pfc_applied += 1;
            if (log_enabled) {
                printSystemPfcState(step, now, ev,
                    pfc_state.paused[ev.priority], dirty.isDirty);
            }
        } else {
            trace.pfc_skipped += 1;
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
    trace.arrival_created = 0;
    trace.arrival_updated = 0;
    trace.arrival_skipped = 0;

    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);

    for (int32_t i = 0; i < inbox.num_arrival; i++) {
        const FlowArrivalEv &ev = inbox.arrivals[i];
        if (ev.port_id != port_id) {
            // Shouldn't happen since deliverEvents dispatched by port_id,
            // but be defensive.
            trace.arrival_skipped += 1;
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
            trace.arrival_updated += 1;
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
            trace.arrival_skipped += 1;
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
    trace.bwupd_created = 0;
    trace.bwupd_updated = 0;
    trace.bwupd_buffered_zero = 0;
    trace.bwupd_destroyed = 0;
    trace.bwupd_forwarded = 0;
    trace.bwupd_completed = 0;
    trace.bwupd_skipped = 0;

    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);

    for (int32_t i = 0; i < inbox.num_bwupd; i++) {
        const BwUpdateEv &ev = inbox.bwupds[i];
        if (ev.port_id != port_id) {
            trace.bwupd_skipped += 1;
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
                    trace.bwupd_buffered_zero += 1;
                    if (log_enabled) {
                        printSystemBwUpdateTag(step, now, "buffered_zero",
                            tag, dirty.isDirty);
                    }
                } else {
                    FlowTagState tag_copy = tag;
                    // Defer destroyTag to flushTagCleanup.
                    if (cleanup.num < MAX_PORT_CLEANUP) {
                        cleanup.tags[cleanup.num] = existing;
                        cleanup.propagate[cleanup.num] = 1;
                        cleanup.num += 1;
                    }
                    // destroyTag sets DirtyPort.isDirty=1 on the owning
                    // port. Mirror that here so the per-tag detail log
                    // line matches the legacy "dirty" field (legacy
                    // printed this AFTER destroyTag ran).
                    dirty.isDirty = 1;
                    trace.bwupd_destroyed += 1;
                    if (log_enabled) {
                        printSystemBwUpdateTag(step, now, "destroy",
                            tag_copy, dirty.isDirty);
                    }
                }
            } else {
                trace.bwupd_skipped += 1;
            }
            continue;
        }

        if (existing == Entity::none()) {
            // Look for cleanup signal for same flow in this port's inbox.
            bool has_cleanup = false;
            for (int32_t j = 0; j < inbox.num_bwupd; j++) {
                if (inbox.bwupds[j].flow_id == ev.flow_id &&
                    inbox.bwupds[j].in_bw == 0.0) {
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
                    if (outbox.num_events < MAX_PORT_OUTBOX) {
                        outbox.events[outbox.num_events++] = cleanup_ev;
                    }
                    trace.bwupd_forwarded += 1;
                    if (log_enabled) {
                        printSystemBwUpdateForward(step, now, ev.flow_id,
                            ev.port_id, next_port);
                    }
                } else {
                    if (completions.num < MAX_PORT_COMPLETE) {
                        completions.flow_ids[completions.num++] = ev.flow_id;
                    }
                    trace.bwupd_completed += 1;
                    if (log_enabled) {
                        printSystemBwUpdateComplete(step, now, ev.flow_id);
                    }
                }
                continue;
            }

            // Defer tag create.
            if (create_list.num < MAX_PORT_CREATE) {
                int32_t pri = 0;
                for (int32_t j = 0; j < numFlowDefs; j++) {
                    if (flowDefs[j].id == ev.flow_id) {
                        pri = flowDefs[j].priority;
                        break;
                    }
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
                trace.bwupd_skipped += 1;
            }
        } else {
            FlowTagState &tag = ctx.get<FlowTagState>(existing);
            if (tag.in_bw != ev.in_bw) {
                materializeBacklog(tag, now);
                tag.in_bw = ev.in_bw;
            }
            dirty.isDirty = 1;
            trace.bwupd_updated += 1;
            if (log_enabled) {
                printSystemBwUpdateTag(step, now, "update", tag, dirty.isDirty);
            }
        }
    }
    inbox.num_bwupd = 0;
}

// Phase E singleton: walks portEntities[] in port_id ascending order and
// materialises each deferred PortCreateReq via createTagOnPort so entity
// ids are allocated in a deterministic order (matching jiuding's
// sequential create order). The per-port arrival/bwUpdate workers may
// not call ctx.makeEntity<FlowTag>() themselves.
void Sim::flushTagCreate(Context &ctx)
{
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortCreateList &cl = ctx.get<PortCreateList>(port_e);
        if (cl.num == 0) {
            continue;
        }
        PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
        for (int32_t i = 0; i < cl.num; i++) {
            const PortCreateReq &req = cl.reqs[i];
            Entity created = createTagOnPort(ctx, port_id, req.flow_id,
                req.in_bw, req.size, req.is_source != 0, req.priority);
            if (created == Entity::none()) {
                if (req.from_arrival != 0) {
                    trace.arrival_skipped += 1;
                } else {
                    trace.bwupd_skipped += 1;
                }
                continue;
            }
            if (req.from_arrival != 0) {
                trace.arrival_created += 1;
            } else {
                trace.bwupd_created += 1;
            }
            if (log_enabled && req.log_enabled != 0) {
                const FlowTagState &tag = ctx.get<FlowTagState>(created);
                int32_t d = ctx.get<DirtyPort>(port_e).isDirty;
                if (req.from_arrival != 0) {
                    printSystemArrivalTag(step, now, req.log_label, tag, d);
                } else {
                    printSystemBwUpdateTag(step, now, req.log_label, tag, d);
                }
            }
        }
        cl.num = 0;
    }
}

// Phase E singleton: apply deferred recordFlowCompletion requests in
// port_id ascending order.
void Sim::flushFlowCompletion(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortCompletionList &cl = ctx.get<PortCompletionList>(port_e);
        for (int32_t i = 0; i < cl.num; i++) {
            recordFlowCompletion(cl.flow_ids[i], now);
        }
        cl.num = 0;
    }
}

// Phase E singleton: emit ingress_chain scope begin/end/summary lines
// once per frame. Per-event detail lines are emitted directly by the
// per-Port workers; canonical sorting in check/run_parity.py tolerates
// inter-port ordering.
void Sim::logIngressChain(Context &ctx)
{
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    if (!log_enabled) {
        // Still clear trace counters even when not logging so next frame
        // starts clean.
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity port_e = portEntities[port_id];
            if (port_e == Entity::none()) continue;
            PortTraceLast &tr = ctx.get<PortTraceLast>(port_e);
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

    int32_t arr_c = 0, arr_u = 0, arr_s = 0;
    int32_t bw_c = 0, bw_u = 0, bw_bz = 0, bw_d = 0, bw_f = 0, bw_cmp = 0, bw_s = 0;
    int32_t pfc_a = 0, pfc_s = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) continue;
        PortTraceLast &tr = ctx.get<PortTraceLast>(port_e);
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

    // Emit the three begin/summary/end blocks in the same order as the
    // legacy singleton path (flow_arrival → bw_update_ingress →
    // pfc_propagate). Strict per-tag detail records were already printed
    // from within the per-Port workers above; canonical-sort in
    // run_parity.py tolerates their interleaving.
    printSystemBegin(step, now, scope, "flow_arrival");
    printSystemArrivalSummary(step, now, arr_c, arr_u, arr_s);
    printSystemEnd(step, now, scope, "flow_arrival");

    printSystemBegin(step, now, scope, "bw_update_ingress");
    printSystemBwUpdateSummary(step, now, bw_c, bw_u, bw_bz, bw_d, bw_f, bw_cmp, bw_s);
    printSystemEnd(step, now, scope, "bw_update_ingress");

    // Legacy pfcPropagate emits its BEGIN/SUMMARY/END block only when
    // enablePfc != 0 (the wrapper guarded the whole system call). Match
    // that so ingress_chain parity holds when PFC is off.
    if (enablePfc != 0) {
        printSystemBegin(step, now, scope, "pfc_propagate");
        printSystemPfcSummary(step, now, pfc_a, pfc_s);
        printSystemEnd(step, now, scope, "pfc_propagate");
    }
}

}
