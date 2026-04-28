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
    FlowId flow_id, int32_t port_id) const
{
    int32_t route_slot = findFlowRouteSlot(flow_id);
    if (route_slot < 0 || route_slot >= numFlowRoutes) {
        return -1;
    }

    const FlowRouteState &route = flowRoutes[route_slot];
    for (int32_t j = 0; j < route.num_steps; j++) {
        if (route.steps[j].port_id == port_id) {
            return route.steps[j].next_port_id;
        }
    }

    return -1;
}

MADRONA_NO_INLINE int32_t Sim::lookupFlowIngressPort(
    FlowId flow_id, int32_t port_id) const
{
    int32_t route_slot = findFlowRouteSlot(flow_id);
    if (route_slot < 0 || route_slot >= numFlowRoutes) {
        return -1;
    }

    const FlowRouteState &route = flowRoutes[route_slot];
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

MADRONA_NO_INLINE void Sim::pushDelayedEvent(const DelayedEvent &ev)
{
    if (numDelayedEvents >= MAX_DELAYED_EVENTS) {
        return;
    }

    int32_t idx = numDelayedEvents++;
    delayedEvents[idx] = ev;

    while (idx > 0 && delayedEvents[idx].t < delayedEvents[idx - 1].t) {
        DelayedEvent tmp = delayedEvents[idx - 1];
        delayedEvents[idx - 1] = delayedEvents[idx];
        delayedEvents[idx] = tmp;
        idx -= 1;
    }
}

MADRONA_NO_INLINE void Sim::pushDelayedEventsBatch(
    const DelayedEvent *events,
    int32_t count)
{
    if (count <= 0) {
        return;
    }

    int32_t available = MAX_DELAYED_EVENTS - numDelayedEvents;
    if (available <= 0) {
        return;
    }
    if (count > available) {
        count = available;
    }

    if (events != delayedEventScratch) {
        for (int32_t i = 0; i < count; i++) {
            delayedEventScratch[i] = events[i];
        }
    }

    if (count > 1) {
        DelayedEvent *src = delayedEventScratch;
        DelayedEvent *dst = delayedEvents + numDelayedEvents;

        for (int32_t width = 1; width < count; width *= 2) {
            for (int32_t left = 0; left < count; left += 2 * width) {
                int32_t mid = minI32(left + width, count);
                int32_t right = minI32(left + 2 * width, count);
                mergeDelayedEventRuns(src, dst, left, mid, right);
            }

            DelayedEvent *tmp = src;
            src = dst;
            dst = tmp;
        }

        if (src != delayedEventScratch) {
            for (int32_t i = 0; i < count; i++) {
                delayedEventScratch[i] = src[i];
            }
        }
    }

    int32_t existing = numDelayedEvents;
    int32_t write = existing + count - 1;
    int32_t i = existing - 1;
    int32_t j = count - 1;

    while (i >= 0 && j >= 0) {
        if (delayedEvents[i].t > delayedEventScratch[j].t) {
            delayedEvents[write--] = delayedEvents[i--];
        } else {
            delayedEvents[write--] = delayedEventScratch[j--];
        }
    }

    while (j >= 0) {
        delayedEvents[write--] = delayedEventScratch[j--];
    }

    numDelayedEvents = existing + count;
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

Time Sim::chooseDT() const
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
        Time gap = delayedEvents[0].t - now;
        if (gap > 1e-15) {
            delayed_gap = gap;
            dt_event = std::min(dt_event, gap);
        }
    }

    if (numPendingFlows > 0) {
        Time gap = pendingFlows[0].start_time - now;
        if (gap > 1e-15) {
            pending_gap = gap;
            dt_event = std::min(dt_event, gap);
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
            if (backlogDrainTimers[port_id] > 1e-15 &&
                timerIsActive(backlogDrainTimers[port_id])) {
                backlog_gap = std::min(backlog_gap,
                    (double)backlogDrainTimers[port_id]);
                dt_event = std::min(dt_event, backlogDrainTimers[port_id]);
            }
        }
    }

    if (enablePfc != 0) {
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            if (pfcPauseTimers[port_id] > 1e-9 &&
                timerIsActive(pfcPauseTimers[port_id])) {
                pfc_pause_gap = std::min(pfc_pause_gap,
                    (double)pfcPauseTimers[port_id]);
                dt_event = std::min(dt_event, pfcPauseTimers[port_id]);
            }
        }
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            if (pfcResumeTimers[port_id] > 1e-9 &&
                timerIsActive(pfcResumeTimers[port_id])) {
                pfc_resume_gap = std::min(pfc_resume_gap,
                    (double)pfcResumeTimers[port_id]);
                dt_event = std::min(dt_event, pfcResumeTimers[port_id]);
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
