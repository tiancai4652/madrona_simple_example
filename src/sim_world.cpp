#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

inline int32_t hashFlowIndex(FlowId flow_id, int32_t count)
{
    if (count <= 0) {
        return 0;
    }

    uint64_t v = (uint64_t)flow_id;
    return (int32_t)(v % (uint64_t)count);
}

inline int32_t minI32(int32_t a, int32_t b)
{
    return a < b ? a : b;
}

MADRONA_NO_INLINE void mergeDelayedEventRuns(
    const DelayedEvent *src,
    DelayedEvent *dst,
    int32_t left,
    int32_t mid,
    int32_t right)
{
    int32_t i = left;
    int32_t j = mid;
    int32_t out = left;

    while (i < mid && j < right) {
        if (src[i].t <= src[j].t) {
            dst[out++] = src[i++];
        } else {
            dst[out++] = src[j++];
        }
    }

    while (i < mid) {
        dst[out++] = src[i++];
    }

    while (j < right) {
        dst[out++] = src[j++];
    }
}

}

int32_t Sim::findNodeSlot(NodeId node_id) const
{
    if (nodeLookupSpan > 0) {
        int64_t lookup_idx = (int64_t)node_id - (int64_t)nodeLookupBase;
        if (lookup_idx >= 0 && lookup_idx < nodeLookupSpan) {
            return nodeSlotLookup[(int32_t)lookup_idx];
        }
    }

    for (int32_t i = 0; i < numTopoNodes; i++) {
        if (topoNodes[i].id == node_id) {
            return i;
        }
    }

    return -1;
}

int32_t Sim::findNeighborSlot(int32_t node_slot, NodeId neighbor_id) const
{
    if (node_slot < 0 || node_slot >= numTopoNodes) {
        return -1;
    }

    const TopoNodeState &node = topoNodes[node_slot];
    for (int32_t i = 0; i < node.num_neighbors; i++) {
        if (node.neighbors[i].neighbor_id == neighbor_id) {
            return i;
        }
    }

    return -1;
}

void Sim::computeRoutes()
{
    for (int32_t i = 0; i < numTopoNodes; i++) {
        for (int32_t j = 0; j < numTopoNodes; j++) {
            routeTable[i][j] = -1;
            ecmpCount[i][j] = 0;
        }
    }

    // loadTopo() expands each physical link into two directed links, so a BFS
    // rooted at the destination can walk the same neighbor lists and recover
    // the shortest-path distance from every source slot to that destination.
    for (int32_t dst_slot = 0; dst_slot < numTopoNodes; dst_slot++) {
        for (int32_t i = 0; i < numTopoNodes; i++) {
            bfsDist[i] = -1;
        }

        int32_t qhead = 0;
        int32_t qtail = 0;
        bfsQueue[qtail++] = dst_slot;
        bfsDist[dst_slot] = 0;

        while (qhead < qtail) {
            int32_t cur_slot = bfsQueue[qhead++];
            const TopoNodeState &cur_node = topoNodes[cur_slot];
            for (int32_t i = 0; i < cur_node.num_neighbors; i++) {
                int32_t neighbor_slot = cur_node.neighbors[i].neighbor_slot;
                if (neighbor_slot < 0 || neighbor_slot >= numTopoNodes) {
                    continue;
                }

                if (bfsDist[neighbor_slot] == -1) {
                    bfsDist[neighbor_slot] = bfsDist[cur_slot] + 1;
                    bfsQueue[qtail++] = neighbor_slot;
                }
            }
        }

        for (int32_t src_slot = 0; src_slot < numTopoNodes; src_slot++) {
            if (src_slot == dst_slot) {
                continue;
            }

            if (topoNodes[src_slot].type != NodeType::Switch) {
                continue;
            }

            int32_t shortest_dist = bfsDist[src_slot];
            if (shortest_dist < 0) {
                continue;
            }

            const TopoNodeState &src_node = topoNodes[src_slot];
            int32_t count = 0;
            for (int32_t i = 0; i < src_node.num_neighbors; i++) {
                const TopoNeighbor &neighbor = src_node.neighbors[i];
                int32_t neighbor_slot = neighbor.neighbor_slot;
                if (neighbor_slot < 0 || neighbor_slot >= numTopoNodes) {
                    continue;
                }

                if (bfsDist[neighbor_slot] >= 0 &&
                    bfsDist[neighbor_slot] + 1 == shortest_dist) {
                    if (count < MAX_ECMP_NEXT_HOPS) {
                        ecmpNextHops[src_slot][dst_slot][count] =
                            neighbor.neighbor_id;
                        count += 1;
                    }
                }
            }

            ecmpCount[src_slot][dst_slot] = count;
            if (count > 0) {
                routeTable[src_slot][dst_slot] =
                    ecmpNextHops[src_slot][dst_slot][0];
            }
        }
    }
}

int32_t Sim::getPath(NodeId src,
                     NodeId dst,
                     FlowId flow_id,
                     NodeId *out_path,
                     int32_t max_path) const
{
    if (max_path <= 0) {
        return 0;
    }

    int32_t count = 0;
    NodeId curr = src;
    out_path[count++] = curr;

    while (curr != dst && count < max_path) {
        int32_t curr_slot = findNodeSlot(curr);
        if (curr_slot < 0) {
            return 0;
        }

        const TopoNodeState &node = topoNodes[curr_slot];
        if (node.type == NodeType::Host) {
            if (node.num_neighbors <= 0) {
                return 0;
            }

            int32_t direct_idx = findNeighborSlot(curr_slot, dst);
            if (direct_idx >= 0) {
                curr = dst;
                out_path[count++] = curr;
                break;
            }

            curr = node.neighbors[0].neighbor_id;
            out_path[count++] = curr;
            continue;
        }

        int32_t dst_slot = findNodeSlot(dst);
        if (dst_slot < 0) {
            return 0;
        }

        int32_t ecmp_num = ecmpCount[curr_slot][dst_slot];
        if (ecmp_num > 0) {
            int32_t idx = hashFlowIndex(flow_id, ecmp_num);
            curr = ecmpNextHops[curr_slot][dst_slot][idx];
            out_path[count++] = curr;
            continue;
        }

        NodeId next_hop = routeTable[curr_slot][dst_slot];
        if (next_hop < 0) {
            return 0;
        }

        curr = next_hop;
        out_path[count++] = curr;
    }

    if (count <= 0 || out_path[count - 1] != dst) {
        return 0;
    }

    return count;
}

MADRONA_NO_INLINE int32_t Sim::lookupFlowRouteNext(
    Context &ctx,
    FlowId flow_id,
    int32_t port_id) const
{
    Entity flow_entity = findFlowMetaEntity(ctx, flow_id);
    if (flow_entity == Entity::none()) {
        return -1;
    }

    const FlowRouteState &route = ctx.get<FlowRouteState>(flow_entity);
    for (int32_t j = 0; j < route.num_steps; j++) {
        if (route.steps[j].port_id == port_id) {
            return route.steps[j].next_port_id;
        }
    }

    return -1;
}

MADRONA_NO_INLINE int32_t Sim::lookupFlowIngressPort(
    Context &ctx,
    FlowId flow_id,
    int32_t port_id) const
{
    Entity flow_entity = findFlowMetaEntity(ctx, flow_id);
    if (flow_entity == Entity::none()) {
        return -1;
    }

    const FlowRouteState &route = ctx.get<FlowRouteState>(flow_entity);
    for (int32_t step_idx = 0; step_idx < route.num_steps; step_idx++) {
        if (route.steps[step_idx].next_port_id != port_id) {
            continue;
        }

        int32_t upstream_port = route.steps[step_idx].port_id;
        if (upstream_port >= 0 && upstream_port < numPorts) {
            return peerPort[upstream_port];
        }
        return -1;
    }

    return -1;
}

namespace {

inline int32_t delayedEventTargetPort(const DelayedEvent &ev)
{
    switch (ev.type) {
    case DelayedEvent::Type::Arrival:
        return ev.arrival.port_id;
    case DelayedEvent::Type::BwUpdate:
        return ev.bwupd.port_id;
    case DelayedEvent::Type::PfcControl:
        return ev.pfcctrl.target_port_id;
    }
    return -1;
}

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

} // namespace

MADRONA_NO_INLINE void Sim::pushDelayedEvent(Context &ctx,
                                             const DelayedEvent &ev)
{
    int32_t target_port = delayedEventTargetPort(ev);
    if (target_port < 0 || target_port >= numPorts) {
        return;
    }

    Entity port_e = portEntities[target_port];
    if (port_e == Entity::none()) {
        return;
    }

    PortDelayedQueue &queue = ctx.get<PortDelayedQueue>(port_e);
    if (queue.head > 0 &&
        queue.head + queue.count >= MAX_PORT_DELAYED_EVENTS) {
        compactPortDelayedQueue(queue);
    }

    if (queue.count >= MAX_PORT_DELAYED_EVENTS) {
        return;
    }

    int32_t idx = queue.head + queue.count;
    queue.count += 1;
    queue.events[idx] = ev;

    while (idx > queue.head && queue.events[idx].t < queue.events[idx - 1].t) {
        DelayedEvent tmp = queue.events[idx - 1];
        queue.events[idx - 1] = queue.events[idx];
        queue.events[idx] = tmp;
        idx -= 1;
    }

    numDelayedEvents += 1;
}

MADRONA_NO_INLINE void Sim::pushDelayedEventsBatch(
    Context &ctx,
    const DelayedEvent *events,
    int32_t count)
{
    if (count <= 0) {
        return;
    }

    for (int32_t i = 0; i < count; i++) {
        pushDelayedEvent(ctx, events[i]);
    }
}

Time Sim::computePropagationTimeAt(Time base_time, Time link_delay) const
{
    Time t = base_time + link_delay;
    if (propagationInterval > 1e-15) {
        Time tick = std::ceil(base_time / propagationInterval) * propagationInterval;
        if (tick < base_time) {
            tick = base_time;
        }
        t = tick + link_delay;
        if (t < base_time) {
            t = base_time + link_delay;
        }
    }
    return t;
}

Time Sim::computePropagationTime(Time link_delay) const
{
    return computePropagationTimeAt(now, link_delay);
}

Time Sim::computePropagationTimeForPort(int32_t src_port_id, int32_t dst_port_id) const
{
    int32_t src_slot = findNodeSlot(portToNode[src_port_id]);
    int32_t dst_slot = findNodeSlot(portToNode[dst_port_id]);
    Time delay = defaultLinkDelay;
    if (src_slot >= 0 && dst_slot >= 0) {
        if (linkDelays[src_slot][dst_slot] >= 0.0) {
            delay = linkDelays[src_slot][dst_slot];
        }
    }
    return computePropagationTime(delay);
}

Time Sim::chooseDT(Context &ctx) const
{
    constexpr const char *scope = "dt";
    uint64_t step = systemLogStep;
    bool log_enabled = compiledSystemLogEnabled(scope, step);

    Time dt_event = std::numeric_limits<Time>::max();
    double delayed_gap = std::numeric_limits<double>::max();
    double pending_gap = std::numeric_limits<double>::max();
    double finish_gap = std::numeric_limits<double>::max();
    double drain_gap = std::numeric_limits<double>::max();
    double backlog_gap = std::numeric_limits<double>::max();
    double pfc_pause_gap = std::numeric_limits<double>::max();
    double pfc_resume_gap = std::numeric_limits<double>::max();

    if (numDelayedEvents > 0) {
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity port_e = portEntities[port_id];
            if (port_e == Entity::none()) {
                continue;
            }
            const PortDelayedQueue &queue = ctx.get<PortDelayedQueue>(port_e);
            if (queue.count <= 0) {
                continue;
            }

            Time gap = queue.events[queue.head].t - now;
            if (gap > 1e-15) {
                delayed_gap = std::min(delayed_gap, (double)gap);
                dt_event = std::min(dt_event, gap);
            }
        }
    }

    const FlowCounters &counters = ctx.singleton<FlowCounters>();

    if (counters.numPendingFlows > 0) {
        Entity flow_entity =
            (counters.pendingFlowCursor >= 0 && network != nullptr &&
             counters.pendingFlowCursor < network->numFlows) ?
            flowMetaEntities[counters.pendingFlowCursor] :
            Entity::none();
        if (flow_entity != Entity::none()) {
            Time gap = ctx.get<FlowDef>(flow_entity).start_time - now;
            if (gap > 1e-15) {
                pending_gap = gap;
                dt_event = std::min(dt_event, gap);
            }
        }
    }

    if (cachedNextFinishTime > 1e-15 &&
        cachedNextFinishTime < timerInactiveSentinel()) {
        finish_gap = cachedNextFinishTime;
        dt_event = std::min(dt_event, cachedNextFinishTime);
    }

    if (enableBuffer != 0 && cachedNextDrainTime > 1e-15 &&
        cachedNextDrainTime < timerInactiveSentinel()) {
        drain_gap = cachedNextDrainTime;
        dt_event = std::min(dt_event, cachedNextDrainTime);
    }

    if (enableBuffer != 0) {
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity port_e = portEntities[port_id];
            if (port_e == Entity::none()) {
                continue;
            }
            Time backlog_drain = ctx.get<PortTimers>(port_e).backlog_drain;
            if (backlog_drain > 1e-15 &&
                timerIsActive(backlog_drain)) {
                backlog_gap = std::min(backlog_gap,
                    (double)backlog_drain);
                dt_event = std::min(dt_event, backlog_drain);
            }
        }
    }

    if (enablePfc != 0) {
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity port_e = portEntities[port_id];
            if (port_e == Entity::none()) {
                continue;
            }
            Time pfc_pause = ctx.get<PortTimers>(port_e).pfc_pause;
            if (pfc_pause > 1e-9 &&
                timerIsActive(pfc_pause)) {
                pfc_pause_gap = std::min(pfc_pause_gap,
                    (double)pfc_pause);
                dt_event = std::min(dt_event, pfc_pause);
            }
        }
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity port_e = portEntities[port_id];
            if (port_e == Entity::none()) {
                continue;
            }
            Time pfc_resume = ctx.get<PortTimers>(port_e).pfc_resume;
            if (pfc_resume > 1e-9 &&
                timerIsActive(pfc_resume)) {
                pfc_resume_gap = std::min(pfc_resume_gap,
                    (double)pfc_resume);
                dt_event = std::min(dt_event, pfc_resume);
            }
        }
    }

    Time dt = dt_event;
    if (dtMin > 0.0 && dt < dtMin) {
        dt = dtMin;
    }
    if (dt > 1e12 || dt == timerInactiveSentinel()) {
        dt = 0.001;
    }

    if (log_enabled) {
        printSystemDTSummary(step, now,
            delayed_gap,
            pending_gap,
            finish_gap,
            drain_gap,
            backlog_gap,
            pfc_pause_gap,
            pfc_resume_gap,
            dt);
    }
    return dt;
}

}
