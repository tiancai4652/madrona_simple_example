#include "sim.hpp"
#include "sim_debug.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

constexpr int32_t MAX_LOGGED_SCHEDULE_FLOWS = 4096;

}

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
    FlowScheduleState schedule_state = FlowScheduleState {};
    if (!prepareFlowScheduleState(flow, schedule_state)) {
        return false;
    }

    FlowCounters &counters = ctx.singleton<FlowCounters>();
    FlowRouteState &route = ctx.get<FlowRouteState>(flow_entity);
    FlowRuntimeState &runtime = ctx.get<FlowRuntimeState>(flow_entity);
    if (schedule_state.port_path_len < 2) {
        return false;
    }

    if (counters.numFlowRoutes < MAX_FLOWS) {
        if (route.num_steps <= 0) {
            counters.numFlowRoutes += 1;
        }
        route = FlowRouteState {};
        route.flow_id = flow.id;
        route.num_steps = schedule_state.port_path_len - 1;
        for (int32_t i = 0; i + 1 < schedule_state.port_path_len; i++) {
            route.steps[i].port_id = schedule_state.port_path[i];
            route.steps[i].next_port_id = schedule_state.port_path[i + 1];
        }
        runtime.route_active = 1;
    }

    out_ev = schedule_state.prepared_event;
    return true;
}

MADRONA_NO_INLINE bool Sim::prepareFlowScheduleState(
    const FlowDef &flow,
    FlowScheduleState &schedule_state) const
{
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

    schedule_state.port_path_len = 0;
    for (int32_t i = 0; i < MAX_FLOW_ROUTE_STEPS + 1; i++) {
        schedule_state.port_path[i] = -1;
    }

    for (int32_t i = 0; i + 1 < path_len; i++) {
        int32_t node_slot = findNodeSlot(path[i]);
        if (node_slot < 0) {
            return false;
        }
        int32_t neighbor_idx = findNeighborSlot(node_slot, path[i + 1]);
        if (neighbor_idx < 0) {
            return false;
        }
        schedule_state.port_path[schedule_state.port_path_len++] =
            topoNodes[node_slot].neighbors[neighbor_idx].port_id;
    }
    schedule_state.port_path[schedule_state.port_path_len++] = -1;

    int32_t src_port_id =
        topoNodes[src_slot].neighbors[first_neighbor_idx].port_id;
    return buildFlowArrivalEvent(src_port_id, flow,
        schedule_state.prepared_event);
}

MADRONA_NO_INLINE void Sim::preparePendingFlowMeta(
    Context &ctx,
    const FlowDef &flow,
    const FlowRuntimeState &runtime,
    FlowScheduleState &schedule_state) const
{
    schedule_state.ready_now = 0;
    schedule_state.prepared = 0;
    schedule_state.port_path_len = 0;
    schedule_state.prepared_event = DelayedEvent {};

    if (runtime.pending == 0) {
        return;
    }

    const FlowCounters &counters = ctx.singleton<FlowCounters>();
    int32_t ready_begin = counters.pendingFlowCursor;
    int32_t pending_end = ready_begin + counters.numPendingFlows;
    if (schedule_state.flow_order < ready_begin ||
        schedule_state.flow_order >= pending_end) {
        return;
    }

    if (flow.start_time > now + 1e-15) {
        return;
    }

    schedule_state.ready_now = 1;
    schedule_state.prepared = prepareFlowScheduleState(flow, schedule_state)
        ? 1
        : 0;
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

        const FlowScheduleState &schedule_state =
            ctx.get<FlowScheduleState>(flow_entity);
        if (schedule_state.ready_now == 0) {
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
            FlowDef logged_flows[MAX_LOGGED_SCHEDULE_FLOWS] {};
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
                const FlowScheduleState &schedule_state =
                    ctx.get<FlowScheduleState>(flow_entity);
                if (flow_runtime.pending == 0) {
                    continue;
                }
                if (log_enabled &&
                    num_logged_flows < MAX_LOGGED_SCHEDULE_FLOWS) {
                    logged_flows[num_logged_flows++] = flow;
                }
                DelayedEvent ev = schedule_state.prepared_event;
                if (schedule_state.prepared != 0 &&
                    sim_runtime.numDelayedEvents < MAX_DELAYED_EVENTS) {
                    pushDelayedEvent(ctx, ev);
                }
                if (schedule_state.prepared != 0 &&
                    counters.numFlowRoutes < MAX_FLOWS) {
                    FlowRouteState &route = ctx.get<FlowRouteState>(flow_entity);
                    if (route.num_steps <= 0) {
                        counters.numFlowRoutes += 1;
                    }
                    route = FlowRouteState {};
                    route.flow_id = flow.id;
                    route.num_steps = schedule_state.port_path_len - 1;
                    for (int32_t j = 0;
                         j + 1 < schedule_state.port_path_len;
                         j++) {
                        route.steps[j].port_id = schedule_state.port_path[j];
                        route.steps[j].next_port_id =
                            schedule_state.port_path[j + 1];
                    }
                    flow_runtime.route_active = 1;
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
        const FlowScheduleState &schedule_state =
            ctx.get<FlowScheduleState>(flow_entity);
        if (flow_runtime.pending == 0) {
            continue;
        }
        if (schedule_state.prepared != 0 &&
            sim_runtime.numDelayedEvents < MAX_DELAYED_EVENTS) {
            pushDelayedEvent(ctx, schedule_state.prepared_event);
        }
        if (schedule_state.prepared != 0 &&
            counters.numFlowRoutes < MAX_FLOWS) {
            const FlowDef &flow = ctx.get<FlowDef>(flow_entity);
            FlowRouteState &route = ctx.get<FlowRouteState>(flow_entity);
            if (route.num_steps <= 0) {
                counters.numFlowRoutes += 1;
            }
            route = FlowRouteState {};
            route.flow_id = flow.id;
            route.num_steps = schedule_state.port_path_len - 1;
            for (int32_t j = 0; j + 1 < schedule_state.port_path_len; j++) {
                route.steps[j].port_id = schedule_state.port_path[j];
                route.steps[j].next_port_id =
                    schedule_state.port_path[j + 1];
            }
            flow_runtime.route_active = 1;
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
