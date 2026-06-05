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
    // Routes are now computed from the compact neighbor lists on demand.
    // This avoids MAX_TOPO_NODES^2 route tables, which are too large for
    // 10k-host fabrics and exceed GPU world-data limits.
}

Time Sim::getLinkDelay(NodeId src, NodeId dst) const
{
    int32_t src_slot = findNodeSlot(src);
    if (src_slot < 0) {
        return defaultLinkDelay;
    }

    int32_t neighbor_idx = findNeighborSlot(src_slot, dst);
    if (neighbor_idx < 0) {
        return defaultLinkDelay;
    }

    Time delay = topoNodes[src_slot].neighbors[neighbor_idx].delay;
    return delay >= 0.0 ? delay : defaultLinkDelay;
}

Time Sim::getPortLinkDelay(int32_t src_port_id, int32_t dst_port_id) const
{
    if (src_port_id < 0 || src_port_id >= numPorts ||
        dst_port_id < 0 || dst_port_id >= numPorts) {
        return defaultLinkDelay;
    }

    return getLinkDelay(portToNode[src_port_id], portToNode[dst_port_id]);
}

namespace {

inline bool appendPathNode(NodeId node, NodeId *out_path,
                           int32_t &count, int32_t max_path)
{
    if (count >= max_path) {
        return false;
    }

    out_path[count++] = node;
    return true;
}

} // namespace

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

    int32_t src_slot = findNodeSlot(src);
    int32_t dst_slot = findNodeSlot(dst);
    if (src_slot < 0 || dst_slot < 0) {
        return 0;
    }

    const TopoNodeState &src_node = topoNodes[src_slot];
    const TopoNodeState &dst_node = topoNodes[dst_slot];
    if (src_node.type == NodeType::Host &&
        dst_node.type == NodeType::Host &&
        src_node.num_neighbors == 1 &&
        dst_node.num_neighbors == 1) {
        NodeId src_leaf = src_node.neighbors[0].neighbor_id;
        NodeId dst_leaf = dst_node.neighbors[0].neighbor_id;
        int32_t src_leaf_slot = src_node.neighbors[0].neighbor_slot;
        int32_t dst_leaf_slot = dst_node.neighbors[0].neighbor_slot;
        if (src_leaf_slot < 0 || dst_leaf_slot < 0) {
            return 0;
        }

        if (!appendPathNode(src_leaf, out_path, count, max_path)) {
            return 0;
        }

        if (src_leaf != dst_leaf) {
            const TopoNodeState &leaf = topoNodes[src_leaf_slot];
            NodeId ecmp_next[MAX_ECMP_NEXT_HOPS] {};
            int32_t ecmp_count = 0;
            for (int32_t i = 0; i < leaf.num_neighbors; i++) {
                const TopoNeighbor &neighbor = leaf.neighbors[i];
                int32_t neigh_slot = neighbor.neighbor_slot;
                if (neigh_slot < 0 || neigh_slot >= numTopoNodes ||
                    topoNodes[neigh_slot].type != NodeType::Switch) {
                    continue;
                }

                if (findNeighborSlot(neigh_slot, dst_leaf) >= 0 &&
                    ecmp_count < MAX_ECMP_NEXT_HOPS) {
                    ecmp_next[ecmp_count++] = neighbor.neighbor_id;
                }
            }

            if (ecmp_count <= 0) {
                return 0;
            }

            int32_t idx = hashFlowIndex(flow_id, ecmp_count);
            if (!appendPathNode(ecmp_next[idx], out_path, count, max_path) ||
                !appendPathNode(dst_leaf, out_path, count, max_path)) {
                return 0;
            }
        }

        if (!appendPathNode(dst, out_path, count, max_path)) {
            return 0;
        }

        return count;
    }

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

        for (int32_t i = 0; i < numTopoNodes; i++) {
            bfsDist[i] = -1;
        }

        int32_t qhead = 0;
        int32_t qtail = 0;
        bfsQueue[qtail++] = dst_slot;
        bfsDist[dst_slot] = 0;

        while (qhead < qtail) {
            int32_t bfs_cur = bfsQueue[qhead++];
            const TopoNodeState &bfs_node = topoNodes[bfs_cur];
            for (int32_t i = 0; i < bfs_node.num_neighbors; i++) {
                int32_t neighbor_slot = bfs_node.neighbors[i].neighbor_slot;
                if (neighbor_slot < 0 || neighbor_slot >= numTopoNodes) {
                    continue;
                }

                if (bfsDist[neighbor_slot] == -1) {
                    bfsDist[neighbor_slot] = bfsDist[bfs_cur] + 1;
                    bfsQueue[qtail++] = neighbor_slot;
                }
            }
        }

        int32_t shortest_dist = bfsDist[curr_slot];
        if (shortest_dist < 0) {
            return 0;
        }

        NodeId next_hops[MAX_ECMP_NEXT_HOPS] {};
        int32_t ecmp_num = 0;
        for (int32_t i = 0; i < node.num_neighbors; i++) {
            const TopoNeighbor &neighbor = node.neighbors[i];
            int32_t neighbor_slot = neighbor.neighbor_slot;
            if (neighbor_slot < 0 || neighbor_slot >= numTopoNodes) {
                continue;
            }

            if (bfsDist[neighbor_slot] >= 0 &&
                bfsDist[neighbor_slot] + 1 == shortest_dist &&
                ecmp_num < MAX_ECMP_NEXT_HOPS) {
                next_hops[ecmp_num++] = neighbor.neighbor_id;
            }
        }

        if (ecmp_num <= 0) {
            return 0;
        }

        curr = next_hops[hashFlowIndex(flow_id, ecmp_num)];
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

    const FlowRuntimeState &runtime =
        ctx.get<FlowRuntimeState>(flow_entity);
    if (runtime.route_active == 0 || runtime.completed != 0) {
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

    const FlowRuntimeState &runtime =
        ctx.get<FlowRuntimeState>(flow_entity);
    if (runtime.route_active == 0 || runtime.completed != 0) {
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

MADRONA_NO_INLINE bool Sim::upstreamTagAlive(
    Context &ctx,
    const FlowTagState &tag) const
{
    if (tag.is_source != 0) {
        return false;
    }

    int32_t upstream_port = -1;
    if (tag.ingress_port_id >= 0 && tag.ingress_port_id < numPorts) {
        upstream_port = peerPort[tag.ingress_port_id];
    }

    if (upstream_port < 0 || upstream_port >= numPorts) {
        return false;
    }

    return findTag(ctx, upstream_port, tag.flow_id) != Entity::none();
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
    SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
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
        runtime.delayedDropCount += 1;
        if (ev.type == DelayedEvent::Type::PfcControl) {
            runtime.delayedPfcDropCount += 1;
        }
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

    runtime.numDelayedEvents += 1;
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
    return computePropagationTime(getPortLinkDelay(src_port_id, dst_port_id));
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
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();

    if (runtime.numDelayedEvents > 0 &&
        runtime.cachedNextDelayedGap > 1e-15 &&
        runtime.cachedNextDelayedGap < timerInactiveSentinel()) {
        delayed_gap = runtime.cachedNextDelayedGap;
        dt_event = std::min(dt_event, runtime.cachedNextDelayedGap);
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

    if (runtime.cachedNextFinishTime > 1e-15 &&
        runtime.cachedNextFinishTime < timerInactiveSentinel()) {
        finish_gap = runtime.cachedNextFinishTime;
        dt_event = std::min(dt_event, runtime.cachedNextFinishTime);
    }

    if (enableBuffer != 0 && runtime.cachedNextDrainTime > 1e-15 &&
        runtime.cachedNextDrainTime < timerInactiveSentinel()) {
        drain_gap = runtime.cachedNextDrainTime;
        dt_event = std::min(dt_event, runtime.cachedNextDrainTime);
    }

    if (enableBuffer != 0 &&
        runtime.cachedNextBacklogGap > 1e-15 &&
        runtime.cachedNextBacklogGap < timerInactiveSentinel()) {
        backlog_gap = runtime.cachedNextBacklogGap;
        dt_event = std::min(dt_event, runtime.cachedNextBacklogGap);
    }

    if (enablePfc != 0) {
        if (runtime.cachedNextPfcPauseGap > 1e-9 &&
            runtime.cachedNextPfcPauseGap < timerInactiveSentinel()) {
            pfc_pause_gap = runtime.cachedNextPfcPauseGap;
            dt_event = std::min(dt_event, runtime.cachedNextPfcPauseGap);
        }
        if (runtime.cachedNextPfcResumeGap > 1e-9 &&
            runtime.cachedNextPfcResumeGap < timerInactiveSentinel()) {
            pfc_resume_gap = runtime.cachedNextPfcResumeGap;
            dt_event = std::min(dt_event, runtime.cachedNextPfcResumeGap);
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
