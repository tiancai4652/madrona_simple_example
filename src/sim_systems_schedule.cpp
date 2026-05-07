#include "sim.hpp"
#include "sim_debug.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

MADRONA_NO_INLINE bool Sim::buildFlowArrivalEvent(int32_t src_port_id,
                                                  const FlowDef &flow,
                                                  DelayedEvent &out_ev) const
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

    out_ev = DelayedEvent {};
    out_ev.t = now;
    out_ev.type = DelayedEvent::Type::Arrival;
    out_ev.arrival = FlowArrivalEv {
        .port_id = src_port_id,
        .flow_id = flow.id,
        .size = flow.size,
        .in_bw = src_in_bw,
        .is_source = 1,
        .priority = flow.priority,
    };
    return true;
}

MADRONA_NO_INLINE bool Sim::injectFlowDef(Context &ctx,
                                          Entity flow_entity,
                                          DelayedEvent &out_ev)
{
    const FlowDef &flow = ctx.get<FlowDef>(flow_entity);
    FlowCounters &counters = ctx.singleton<FlowCounters>();
    NodeId path[MAX_PATH_NODES] {};
    int32_t path_len = getPath(flow.src_node, flow.dst_node, flow.id,
        path, MAX_PATH_NODES);
    if (path_len < 2) {
        return false;
    }

    int32_t src_slot = findNodeSlot(flow.src_node);
    if (src_slot < 0) {
        return false;
    }

    NodeId first_hop = path[1];
    int32_t first_neighbor_idx = findNeighborSlot(src_slot, first_hop);
    if (first_neighbor_idx < 0) {
        return false;
    }

    int32_t port_path[MAX_FLOW_ROUTE_STEPS + 1] {};
    int32_t port_path_len = 0;

    for (int32_t i = 0; i + 1 < path_len; i++) {
        int32_t node_slot = findNodeSlot(path[i]);
        if (node_slot < 0) {
            return false;
        }
        int32_t neighbor_idx = findNeighborSlot(node_slot, path[i + 1]);
        if (neighbor_idx < 0) {
            return false;
        }
        port_path[port_path_len++] =
            topoNodes[node_slot].neighbors[neighbor_idx].port_id;
    }
    port_path[port_path_len++] = -1;

    if (counters.numFlowRoutes < MAX_FLOWS) {
        FlowRouteState &route = ctx.get<FlowRouteState>(flow_entity);
        FlowRuntimeState &runtime = ctx.get<FlowRuntimeState>(flow_entity);
        if (route.num_steps <= 0) {
            counters.numFlowRoutes += 1;
        }
        route = FlowRouteState {};
        route.flow_id = flow.id;
        route.num_steps = port_path_len - 1;
        for (int32_t i = 0; i + 1 < port_path_len; i++) {
            route.steps[i].port_id = port_path[i];
            route.steps[i].next_port_id = port_path[i + 1];
        }
        runtime.route_active = 1;
    }

    int32_t src_port_id =
        topoNodes[src_slot].neighbors[first_neighbor_idx].port_id;
    return buildFlowArrivalEvent(src_port_id, flow, out_ev);
}

MADRONA_NO_INLINE void Sim::schedulePendingFlows(Context &ctx)
{
    FlowCounters &counters = ctx.singleton<FlowCounters>();
    SimRuntimeState &sim_runtime = ctx.singleton<SimRuntimeState>();
    int32_t pending_end =
        counters.pendingFlowCursor + counters.numPendingFlows;
    int32_t ready_begin = counters.pendingFlowCursor;
    int32_t ready_end = ready_begin;
    while (ready_end < pending_end) {
        Entity flow_entity = flowMetaEntities[ready_end];
        if (flow_entity == Entity::none()) {
            ready_end += 1;
            continue;
        }

        FlowRuntimeState &runtime = ctx.get<FlowRuntimeState>(flow_entity);
        if (runtime.pending == 0) {
            ready_end += 1;
            continue;
        }

        if (ctx.get<FlowDef>(flow_entity).start_time > now + 1e-15) {
            break;
        }

        ready_end += 1;
    }
    int32_t ready_count = ready_end - ready_begin;

    if constexpr (system_log_compiled_in) {
        if (traceModeEnabled()) {
            int32_t pending_before = counters.numPendingFlows;
            int32_t delayed_before = sim_runtime.numDelayedEvents;
            int32_t flow_routes_before = counters.numFlowRoutes;
            constexpr const char *scope = "ingress_chain";
            uint64_t step = systemLogStep;
            bool log_enabled = compiledSystemLogEnabled(scope, step);
            FlowDef logged_flows[MAX_FLOWS] {};
            int32_t num_logged_flows = 0;

            if (log_enabled) {
                printSystemBegin(step, now, scope, "schedule_pending_flows");
            }

            for (int32_t i = ready_begin; i < ready_end; i++) {
                Entity flow_entity = flowMetaEntities[i];
                if (flow_entity == Entity::none()) {
                    continue;
                }
                const FlowDef &flow = ctx.get<FlowDef>(flow_entity);
                FlowRuntimeState &flow_runtime = ctx.get<FlowRuntimeState>(
                    flow_entity);
                if (flow_runtime.pending == 0) {
                    continue;
                }
                if (log_enabled && num_logged_flows < MAX_FLOWS) {
                    logged_flows[num_logged_flows++] = flow;
                }
                DelayedEvent ev {};
                if (injectFlowDef(ctx, flow_entity, ev) &&
                    sim_runtime.numDelayedEvents < MAX_DELAYED_EVENTS) {
                    pushDelayedEvent(ctx, ev);
                }
                flow_runtime.pending = 0;
                flow_runtime.active = 1;
            }

            if (ready_count > 0) {
                counters.pendingFlowCursor = ready_end;
                int32_t remaining =
                    pending_end - counters.pendingFlowCursor;
                if (remaining < 0) {
                    remaining = 0;
                }
                counters.numPendingFlows = remaining;
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
                    counters.numPendingFlows,
                    delayed_before,
                    sim_runtime.numDelayedEvents,
                    flow_routes_before,
                    counters.numFlowRoutes);
                printSystemEnd(step, now, scope, "schedule_pending_flows");
            }
            return;
        }
    }

    for (int32_t i = ready_begin; i < ready_end; i++) {
        Entity flow_entity = flowMetaEntities[i];
        if (flow_entity == Entity::none()) {
            continue;
        }
        FlowRuntimeState &flow_runtime = ctx.get<FlowRuntimeState>(flow_entity);
        if (flow_runtime.pending == 0) {
            continue;
        }
        DelayedEvent ev {};
        if (injectFlowDef(ctx, flow_entity, ev) &&
            sim_runtime.numDelayedEvents < MAX_DELAYED_EVENTS) {
            pushDelayedEvent(ctx, ev);
        }
        flow_runtime.pending = 0;
        flow_runtime.active = 1;
    }

    if (ready_count > 0) {
        counters.pendingFlowCursor = ready_end;
        int32_t remaining = pending_end - counters.pendingFlowCursor;
        if (remaining < 0) {
            remaining = 0;
        }
        counters.numPendingFlows = remaining;
    }
}

}
