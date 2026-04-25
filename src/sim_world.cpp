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

}

int32_t Sim::findNodeSlot(NodeId node_id) const
{
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
    // BFS scratch buffers (bfsAdjCount/bfsAdj/bfsDist/bfsQueue) live on the Sim
    // instance to avoid the ~10 MB stack footprint that NxN local arrays would
    // produce at MAX_TOPO_NODES = 1152.
    for (int32_t i = 0; i < MAX_TOPO_NODES; i++) {
        bfsAdjCount[i] = 0;
        for (int32_t j = 0; j < MAX_TOPO_NODES; j++) {
            bfsAdj[i][j] = -1;
        }
    }

    for (int32_t i = 0; i < numTopoLinks; i++) {
        int32_t src_slot = findNodeSlot(topoLinks[i].src);
        if (src_slot < 0) {
            continue;
        }

        int32_t idx = bfsAdjCount[src_slot]++;
        bfsAdj[src_slot][idx] = topoLinks[i].dst;
    }

    for (int32_t i = 0; i < MAX_TOPO_NODES; i++) {
        for (int32_t j = 0; j < MAX_TOPO_NODES; j++) {
            bfsDist[i][j] = -1;
            routeTable[i][j] = -1;
            ecmpCount[i][j] = 0;
            for (int32_t k = 0; k < MAX_ECMP_NEXT_HOPS; k++) {
                ecmpNextHops[i][j][k] = -1;
            }
        }
    }

    for (int32_t src_slot = 0; src_slot < numTopoNodes; src_slot++) {
        int32_t qhead = 0;
        int32_t qtail = 0;
        bfsQueue[qtail++] = src_slot;
        bfsDist[src_slot][src_slot] = 0;

        while (qhead < qtail) {
            int32_t u_slot = bfsQueue[qhead++];
            for (int32_t i = 0; i < bfsAdjCount[u_slot]; i++) {
                NodeId v_id = bfsAdj[u_slot][i];
                int32_t v_slot = findNodeSlot(v_id);
                if (v_slot < 0) {
                    continue;
                }
                if (bfsDist[src_slot][v_slot] == -1) {
                    bfsDist[src_slot][v_slot] = bfsDist[src_slot][u_slot] + 1;
                    bfsQueue[qtail++] = v_slot;
                }
            }
        }
    }

    for (int32_t src_slot = 0; src_slot < numTopoNodes; src_slot++) {
        if (topoNodes[src_slot].type != NodeType::Switch) {
            continue;
        }

        for (int32_t dst_slot = 0; dst_slot < numTopoNodes; dst_slot++) {
            if (src_slot == dst_slot) {
                continue;
            }

            int32_t shortest_dist = bfsDist[src_slot][dst_slot];
            if (shortest_dist < 0) {
                continue;
            }

            int32_t count = 0;
            for (int32_t i = 0; i < bfsAdjCount[src_slot]; i++) {
                NodeId neighbor_id = bfsAdj[src_slot][i];
                int32_t neighbor_slot = findNodeSlot(neighbor_id);
                if (neighbor_slot < 0) {
                    continue;
                }

                if (bfsDist[neighbor_slot][dst_slot] >= 0 &&
                    1 + bfsDist[neighbor_slot][dst_slot] == shortest_dist) {
                    if (count < MAX_ECMP_NEXT_HOPS) {
                        ecmpNextHops[src_slot][dst_slot][count] = neighbor_id;
                        count += 1;
                    }
                }
            }

            ecmpCount[src_slot][dst_slot] = count;
            if (count > 0) {
                routeTable[src_slot][dst_slot] = ecmpNextHops[src_slot][dst_slot][0];
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

int32_t Sim::lookupFlowRouteNext(FlowId flow_id, int32_t port_id) const
{
    for (int32_t i = 0; i < numFlowRoutes; i++) {
        if (flowRoutes[i].flow_id != flow_id) {
            continue;
        }

        for (int32_t j = 0; j < flowRoutes[i].num_steps; j++) {
            if (flowRoutes[i].steps[j].port_id == port_id) {
                return flowRoutes[i].steps[j].next_port_id;
            }
        }
    }

    return -1;
}

void Sim::pushDelayedEvent(const DelayedEvent &ev)
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
    bool log_enabled = systemLogEnabled(scope, step);

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

    if (cachedNextFinishTime > 1e-15 && cachedNextFinishTime < std::numeric_limits<Time>::max()) {
        finish_gap = cachedNextFinishTime;
        dt_event = std::min(dt_event, cachedNextFinishTime);
    }

    if (enableBuffer != 0 && cachedNextDrainTime > 1e-15 && cachedNextDrainTime < std::numeric_limits<Time>::max()) {
        drain_gap = cachedNextDrainTime;
        dt_event = std::min(dt_event, cachedNextDrainTime);
    }

    if (enableBuffer != 0) {
        for (int32_t i = 0; i < numBacklogDrainTimers; i++) {
            if (backlogDrainTimers[i] > 1e-15) {
                backlog_gap = std::min(backlog_gap, (double)backlogDrainTimers[i]);
                dt_event = std::min(dt_event, backlogDrainTimers[i]);
            }
        }
    }

    if (enablePfc != 0) {
        for (int32_t i = 0; i < numPfcPauseTimers; i++) {
            if (pfcPauseTimers[i] > 1e-9) {
                pfc_pause_gap = std::min(pfc_pause_gap, (double)pfcPauseTimers[i]);
                dt_event = std::min(dt_event, pfcPauseTimers[i]);
            }
        }
        for (int32_t i = 0; i < numPfcResumeTimers; i++) {
            if (pfcResumeTimers[i] > 1e-9) {
                pfc_resume_gap = std::min(pfc_resume_gap, (double)pfcResumeTimers[i]);
                dt_event = std::min(dt_event, pfcResumeTimers[i]);
            }
        }
    }

    Time dt = dt_event;
    if (dtMin > 0.0 && dt < dtMin) {
        dt = dtMin;
    }
    if (dt > 1e12 || dt == std::numeric_limits<Time>::max()) {
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
