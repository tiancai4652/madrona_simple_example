#include "sim.hpp"
#include "sim_debug.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

MADRONA_NO_INLINE void Sim::injectFlow(int32_t src_port_id,
                                       const FlowDef &flow)
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

MADRONA_NO_INLINE void Sim::injectFlowDef(const FlowDef &flow)
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
        port_path[port_path_len++] =
            topoNodes[node_slot].neighbors[neighbor_idx].port_id;
    }
    port_path[port_path_len++] = -1;

    // schedulePendingFlows consumes each FlowDef at most once, so route
    // insertion is append-only during startup and does not need an O(N)
    // duplicate scan per flow on the GPU path.
    if (numFlowRoutes < MAX_FLOWS) {
        FlowRouteState &route = flowRoutes[numFlowRoutes++];
        route.flow_id = flow.id;
        route.num_steps = port_path_len - 1;
        for (int32_t i = 0; i + 1 < port_path_len; i++) {
            route.steps[i].port_id = port_path[i];
            route.steps[i].next_port_id = port_path[i + 1];
        }
    }

    int32_t src_port_id =
        topoNodes[src_slot].neighbors[first_neighbor_idx].port_id;
    injectFlow(src_port_id, flow);
}

MADRONA_NO_INLINE void Sim::schedulePendingFlows()
{
    int32_t ready_count = 0;
    while (ready_count < numPendingFlows &&
           pendingFlows[ready_count].start_time <= now + 1e-15) {
        ready_count += 1;
    }

    if constexpr (system_log_compiled_in) {
        int32_t pending_before = numPendingFlows;
        int32_t delayed_before = numDelayedEvents;
        int32_t flow_routes_before = numFlowRoutes;
        constexpr const char *scope = "ingress_chain";
        uint64_t step = systemLogStep;
        bool log_enabled = compiledSystemLogEnabled(scope, step);
        FlowDef logged_flows[MAX_FLOWS] {};
        int32_t num_logged_flows = 0;

        if (log_enabled) {
            printSystemBegin(step, now, scope, "schedule_pending_flows");
        }

        for (int32_t i = 0; i < ready_count; i++) {
            FlowDef flow = pendingFlows[i];
            if (log_enabled && num_logged_flows < MAX_FLOWS) {
                logged_flows[num_logged_flows++] = flow;
            }
            injectFlowDef(flow);
        }

        if (ready_count > 0) {
            for (int32_t i = ready_count; i < numPendingFlows; i++) {
                pendingFlows[i - ready_count] = pendingFlows[i];
            }
            numPendingFlows -= ready_count;
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
                ready_count,
                pending_before,
                numPendingFlows,
                delayed_before,
                numDelayedEvents,
                flow_routes_before,
                numFlowRoutes);
            printSystemEnd(step, now, scope, "schedule_pending_flows");
        }
    } else {
        for (int32_t i = 0; i < ready_count; i++) {
            injectFlowDef(pendingFlows[i]);
        }

        if (ready_count > 0) {
            for (int32_t i = ready_count; i < numPendingFlows; i++) {
                pendingFlows[i - ready_count] = pendingFlows[i];
            }
            numPendingFlows -= ready_count;
        }
    }
}

}
