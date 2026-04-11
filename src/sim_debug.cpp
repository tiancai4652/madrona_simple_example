#include "sim_debug.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace madsimple {

namespace {

#if defined(__CUDA_ARCH__)
inline const char *nodeTypeName(NodeType)
{
    return "host";
}

inline void printNodeList(const NodeId *, int32_t)
{}
#else
inline const char *nodeTypeName(NodeType type)
{
    return type == NodeType::Host ? "host" : "switch";
}

inline void printNodeList(const NodeId *nodes, int32_t count)
{
    std::cout << '[';
    for (int32_t i = 0; i < count; i++) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << nodes[i];
    }
    std::cout << ']';
}

inline bool scopeMatches(const char *want, const char *actual)
{
    return std::strcmp(want, actual) == 0;
}
#endif

}

#if defined(__CUDA_ARCH__)
const bool init_log_print_enabled = false;
const bool system_log_print_enabled = false;

bool systemLogEnabled(const char *, uint64_t)
{
    return false;
}

void printInitTopoLog(const Sim &, Engine &) {}
void printInitFlowLog(const Sim &) {}
void printSystemBegin(uint64_t, Time, const char *, const char *) {}
void printSystemEnd(uint64_t, Time, const char *, const char *) {}
void printSystemScheduleFlow(uint64_t, Time, const FlowDef &) {}
void printSystemScheduleSummary(uint64_t, Time, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t) {}
void printSystemDeliverArrival(uint64_t, Time, const FlowArrivalEv &) {}
void printSystemDeliverBwUpdate(uint64_t, Time, const BwUpdateEv &) {}
void printSystemDeliverPfc(uint64_t, Time, const PfcControlEv &) {}
void printSystemDeliverSummary(uint64_t, Time, int32_t, int32_t, int32_t, int32_t, int32_t) {}
void printSystemArrivalTag(uint64_t, Time, const char *, const FlowTagState &, int32_t) {}
void printSystemArrivalSummary(uint64_t, Time, int32_t, int32_t, int32_t) {}
void printSystemBwUpdateTag(uint64_t, Time, const char *, const FlowTagState &, int32_t) {}
void printSystemBwUpdateForward(uint64_t, Time, FlowId, int32_t, int32_t) {}
void printSystemBwUpdateComplete(uint64_t, Time, FlowId) {}
void printSystemBwUpdateSummary(uint64_t, Time, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t) {}
void printSystemPfcState(uint64_t, Time, const PfcControlEv &, int32_t, int32_t) {}
void printSystemPfcSummary(uint64_t, Time, int32_t, int32_t) {}
void printSystemAllocPort(uint64_t, Time, int32_t, double, int32_t, int32_t, double, double, int32_t, int32_t, int32_t) {}
void printSystemAllocSummary(uint64_t, Time, int32_t, int32_t, int32_t) {}
void printSystemEmitSummary(uint64_t, Time, int32_t, int32_t, int32_t) {}
void printSystemPfcDetectSummary(uint64_t, Time, int32_t, int32_t, int32_t, int32_t) {}
void printSystemClearSummary(uint64_t, Time, int32_t) {}
void printSystemDTSummary(uint64_t, Time, double, double, double, double, double, double, double, double) {}
void printSystemBufferSummary(uint64_t, Time, int32_t, int32_t, double) {}
#else
const bool init_log_print_enabled = []() {
    const char *env = std::getenv("init_log_print_enabled");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }

    return !(env[0] == '0' && env[1] == '\0');
}();

const bool system_log_print_enabled = []() {
    const char *env = std::getenv("system_log_print_enabled");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }

    return !(env[0] == '0' && env[1] == '\0');
}();

bool systemLogEnabled(const char *scope, uint64_t step)
{
    if (!system_log_print_enabled) {
        return false;
    }

    const char *scope_env = std::getenv("system_log_scope");
    if (scope_env != nullptr && scope_env[0] != '\0' &&
        !scopeMatches(scope_env, "all") && !scopeMatches(scope_env, scope)) {
        return false;
    }

    const char *step_env = std::getenv("system_log_step");
    if (step_env != nullptr && step_env[0] != '\0') {
        char *end = nullptr;
        unsigned long want = std::strtoul(step_env, &end, 10);
        if (end != step_env && *end == '\0' && step != want) {
            return false;
        }
    }

    return true;
}

void printInitTopoLog(const Sim &sim, Engine &ctx)
{
    std::cout << std::fixed << std::setprecision(6);

    int32_t host_count = 0;
    int32_t switch_count = 0;
    for (int32_t i = 0; i < sim.numTopoNodes; i++) {
        if (sim.topoNodes[i].type == NodeType::Host) {
            host_count += 1;
        } else {
            switch_count += 1;
        }
    }

    std::cout << "[INIT][TOPO] num_nodes=" << sim.numTopoNodes
              << " num_hosts=" << host_count
              << " num_switches=" << switch_count
              << " num_bidirectional_links=" << (sim.numTopoLinks / 2)
              << " num_directional_ports=" << sim.numPorts
              << " port_entities_count=" << sim.numPorts
              << " flow_tag_entities_count=0\n";

    for (int32_t i = 0; i < sim.numTopoNodes; i++) {
        const TopoNodeState &node = sim.topoNodes[i];
        std::cout << "[INIT][TOPO][NODE] node_id=" << node.id
                  << " node_type=" << nodeTypeName(node.type)
                  << " port_bw=" << node.port_bw
                  << " num_neighbors=" << node.num_neighbors
                  << "\n";
    }

    for (int32_t port_id = 0; port_id < sim.numPorts; port_id++) {
        NodeId neighbor_id = -1;
        for (int32_t i = 0; i < sim.numTopoNodes; i++) {
            const TopoNodeState &node = sim.topoNodes[i];
            for (int32_t j = 0; j < node.num_neighbors; j++) {
                if (node.neighbors[j].port_id == port_id) {
                    neighbor_id = node.neighbors[j].neighbor_id;
                    break;
                }
            }
            if (neighbor_id >= 0) {
                break;
            }
        }

        const PortState &port = ctx.get<PortState>(sim.portEntities[port_id]);
        std::cout << "[INIT][TOPO][PORT] port_id=" << port.port_id
                  << " node_id=" << port.node_id
                  << " neighbor_id=" << neighbor_id
                  << " port_idx=" << port.port_idx
                  << " port_bw=" << port.port_bw
                  << " connected=" << port.connected
                  << " peer_port=" << sim.peerPort[port_id]
                  << "\n";
    }

    NodeId node_ids[MAX_TOPO_NODES] {};
    for (int32_t i = 0; i < sim.numTopoNodes; i++) {
        node_ids[i] = sim.topoNodes[i].id;
    }
    for (int32_t i = 0; i < sim.numTopoNodes; i++) {
        for (int32_t j = i + 1; j < sim.numTopoNodes; j++) {
            if (node_ids[j] < node_ids[i]) {
                NodeId tmp = node_ids[i];
                node_ids[i] = node_ids[j];
                node_ids[j] = tmp;
            }
        }
    }

    for (int32_t i = 0; i < sim.numTopoNodes; i++) {
        for (int32_t j = 0; j < sim.numTopoNodes; j++) {
            NodeId src = node_ids[i];
            NodeId dst = node_ids[j];
            if (src == dst) {
                continue;
            }

            int32_t src_slot = sim.findNodeSlot(src);
            int32_t dst_slot = sim.findNodeSlot(dst);
            NodeId route = -1;
            int32_t ecmp_count = 0;
            NodeId sorted_ecmp[MAX_ECMP_NEXT_HOPS] {};
            if (src_slot >= 0 && dst_slot >= 0) {
                route = sim.routeTable[src_slot][dst_slot];
                ecmp_count = sim.ecmpCount[src_slot][dst_slot];
                for (int32_t k = 0; k < ecmp_count; k++) {
                    sorted_ecmp[k] = sim.ecmpNextHops[src_slot][dst_slot][k];
                }
                for (int32_t a = 0; a < ecmp_count; a++) {
                    for (int32_t b = a + 1; b < ecmp_count; b++) {
                        if (sorted_ecmp[b] < sorted_ecmp[a]) {
                            NodeId tmp = sorted_ecmp[a];
                            sorted_ecmp[a] = sorted_ecmp[b];
                            sorted_ecmp[b] = tmp;
                        }
                    }
                }
            }

            std::cout << "[INIT][ROUTE] src=" << src
                      << " dst=" << dst
                      << " route=" << route
                      << " ecmp_count=" << ecmp_count
                      << " ecmp=";
            if (src_slot >= 0 && dst_slot >= 0) {
                printNodeList(sorted_ecmp, ecmp_count);
            } else {
                std::cout << "[]";
            }
            std::cout << "\n";
        }
    }
}

void printInitFlowLog(const Sim &sim)
{
    std::cout << std::fixed << std::setprecision(6);

    int32_t pending_sorted = 1;
    for (int32_t i = 1; i < sim.numPendingFlows; i++) {
        if (sim.pendingFlows[i - 1].start_time > sim.pendingFlows[i].start_time) {
            pending_sorted = 0;
            break;
        }
    }

    std::cout << "[INIT][FLOW] num_flow_defs=" << sim.numFlowDefs
              << " num_pending_flows=" << sim.numPendingFlows
              << " pending_sorted_by_start_time=" << pending_sorted
              << "\n";

    FlowDef sorted_defs[MAX_FLOWS] {};
    for (int32_t i = 0; i < sim.numFlowDefs; i++) {
        sorted_defs[i] = sim.flowDefs[i];
    }
    for (int32_t i = 0; i < sim.numFlowDefs; i++) {
        for (int32_t j = i + 1; j < sim.numFlowDefs; j++) {
            if (sorted_defs[j].id < sorted_defs[i].id) {
                FlowDef tmp = sorted_defs[i];
                sorted_defs[i] = sorted_defs[j];
                sorted_defs[j] = tmp;
            }
        }
    }

    int32_t num_print = sim.numFlowDefs < 20 ? sim.numFlowDefs : 20;
    for (int32_t i = 0; i < num_print; i++) {
        const FlowDef &flow = sorted_defs[i];
        std::cout << "[INIT][FLOW][DEF] flow_id=" << flow.id
                  << " src_node=" << flow.src_node
                  << " dst_node=" << flow.dst_node
                  << " size=" << flow.size
                  << " start_time=" << flow.start_time
                  << " priority=" << flow.priority
                  << "\n";
    }

    std::cout << "[INIT][FLOW][STATE] pending_flows_count=" << sim.numPendingFlows
              << " source_tags_count=" << sim.numSourceTags
              << " flow_routes_count=" << sim.numFlowRoutes
              << " flow_routes_state="
              << (sim.numFlowRoutes == 0 ? "empty" : "pre_generated")
              << " flow_tag_entities_count=0\n";
}

void printSystemBegin(uint64_t step, Time now, const char *scope, const char *phase)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BEGIN] step=" << step
              << " now=" << now
              << " scope=" << scope
              << " phase=" << phase
              << "\n";
}

void printSystemEnd(uint64_t step, Time now, const char *scope, const char *phase)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][END] step=" << step
              << " now=" << now
              << " scope=" << scope
              << " phase=" << phase
              << "\n";
}

void printSystemScheduleFlow(uint64_t step, Time now, const FlowDef &flow)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][SCHEDULE][FLOW] step=" << step
              << " now=" << now
              << " flow_id=" << flow.id
              << " src_node=" << flow.src_node
              << " dst_node=" << flow.dst_node
              << " size=" << flow.size
              << " start_time=" << flow.start_time
              << " priority=" << flow.priority
              << "\n";
}

void printSystemScheduleSummary(uint64_t step, Time now,
                                int32_t scheduled_count,
                                int32_t pending_before,
                                int32_t pending_after,
                                int32_t delayed_before,
                                int32_t delayed_after,
                                int32_t flow_routes_before,
                                int32_t flow_routes_after)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][SCHEDULE][SUMMARY] step=" << step
              << " now=" << now
              << " scheduled_count=" << scheduled_count
              << " pending_before=" << pending_before
              << " pending_after=" << pending_after
              << " delayed_before=" << delayed_before
              << " delayed_after=" << delayed_after
              << " flow_routes_before=" << flow_routes_before
              << " flow_routes_after=" << flow_routes_after
              << "\n";
}

void printSystemDeliverArrival(uint64_t step, Time now, const FlowArrivalEv &ev)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][DELIVER][EVENT] step=" << step
              << " now=" << now
              << " event_type=arrival"
              << " port_id=" << ev.port_id
              << " flow_id=" << ev.flow_id
              << " in_bw=" << ev.in_bw
              << " size=" << ev.size
              << " is_source=" << ev.is_source
              << " priority=" << ev.priority
              << "\n";
}

void printSystemDeliverBwUpdate(uint64_t step, Time now, const BwUpdateEv &ev)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][DELIVER][EVENT] step=" << step
              << " now=" << now
              << " event_type=bwupdate"
              << " port_id=" << ev.port_id
              << " flow_id=" << ev.flow_id
              << " in_bw=" << ev.in_bw
              << "\n";
}

void printSystemDeliverPfc(uint64_t step, Time now, const PfcControlEv &ev)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][DELIVER][EVENT] step=" << step
              << " now=" << now
              << " event_type=pfc"
              << " target_port_id=" << ev.target_port_id
              << " source_port_id=" << ev.source_port_id
              << " priority=" << ev.priority
              << " paused=" << ev.paused
              << "\n";
}

void printSystemDeliverSummary(uint64_t step, Time now,
                               int32_t delayed_before,
                               int32_t delayed_after,
                               int32_t inbox_arrival_count,
                               int32_t inbox_bwupdate_count,
                               int32_t inbox_pfc_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][DELIVER][SUMMARY] step=" << step
              << " now=" << now
              << " delayed_before=" << delayed_before
              << " delayed_after=" << delayed_after
              << " inbox_arrival_count=" << inbox_arrival_count
              << " inbox_bwupdate_count=" << inbox_bwupdate_count
              << " inbox_pfc_count=" << inbox_pfc_count
              << "\n";
}

void printSystemArrivalTag(uint64_t step, Time now, const char *action,
                           const FlowTagState &tag, int32_t dirty)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][ARRIVAL][TAG] step=" << step
              << " now=" << now
              << " action=" << action
              << " port_id=" << tag.port_id
              << " flow_id=" << tag.flow_id
              << " in_bw=" << tag.in_bw
              << " remaining=" << tag.remaining
              << " is_source=" << tag.is_source
              << " priority=" << tag.priority
              << " next_port_id=" << tag.next_port_id
              << " ingress_port_id=" << tag.ingress_port_id
              << " dirty=" << dirty
              << "\n";
}

void printSystemArrivalSummary(uint64_t step, Time now,
                               int32_t created_count,
                               int32_t updated_count,
                               int32_t skipped_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][ARRIVAL][SUMMARY] step=" << step
              << " now=" << now
              << " created_count=" << created_count
              << " updated_count=" << updated_count
              << " skipped_count=" << skipped_count
              << "\n";
}

void printSystemBwUpdateTag(uint64_t step, Time now, const char *action,
                            const FlowTagState &tag, int32_t dirty)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BWUPDATE][TAG] step=" << step
              << " now=" << now
              << " action=" << action
              << " port_id=" << tag.port_id
              << " flow_id=" << tag.flow_id
              << " in_bw=" << tag.in_bw
              << " out_bw=" << tag.out_bw
              << " backlog=" << tag.backlog
              << " priority=" << tag.priority
              << " is_source=" << tag.is_source
              << " next_port_id=" << tag.next_port_id
              << " ingress_port_id=" << tag.ingress_port_id
              << " dirty=" << dirty
              << "\n";
}

void printSystemBwUpdateForward(uint64_t step, Time now,
                                FlowId flow_id,
                                int32_t port_id,
                                int32_t next_port_id)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BWUPDATE][FORWARD] step=" << step
              << " now=" << now
              << " flow_id=" << flow_id
              << " port_id=" << port_id
              << " next_port_id=" << next_port_id
              << "\n";
}

void printSystemBwUpdateComplete(uint64_t step, Time now, FlowId flow_id)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BWUPDATE][COMPLETE] step=" << step
              << " now=" << now
              << " flow_id=" << flow_id
              << "\n";
}

void printSystemBwUpdateSummary(uint64_t step, Time now,
                                int32_t created_count,
                                int32_t updated_count,
                                int32_t buffered_zero_count,
                                int32_t destroyed_count,
                                int32_t forwarded_count,
                                int32_t completed_count,
                                int32_t skipped_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BWUPDATE][SUMMARY] step=" << step
              << " now=" << now
              << " created_count=" << created_count
              << " updated_count=" << updated_count
              << " buffered_zero_count=" << buffered_zero_count
              << " destroyed_count=" << destroyed_count
              << " forwarded_count=" << forwarded_count
              << " completed_count=" << completed_count
              << " skipped_count=" << skipped_count
              << "\n";
}

void printSystemPfcState(uint64_t step, Time now, const PfcControlEv &ev,
                         int32_t applied_paused, int32_t dirty)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][PFC][STATE] step=" << step
              << " now=" << now
              << " target_port_id=" << ev.target_port_id
              << " source_port_id=" << ev.source_port_id
              << " priority=" << ev.priority
              << " paused=" << ev.paused
              << " applied_paused=" << applied_paused
              << " dirty=" << dirty
              << "\n";
}

void printSystemPfcSummary(uint64_t step, Time now,
                           int32_t applied_count,
                           int32_t skipped_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][PFC][SUMMARY] step=" << step
              << " now=" << now
              << " applied_count=" << applied_count
              << " skipped_count=" << skipped_count
              << "\n";
}

void printSystemAllocPort(uint64_t step, Time now,
                          int32_t port_id,
                          double port_bw,
                          int32_t num_tags,
                          int32_t num_live,
                          double sum_in,
                          double sum_out,
                          int32_t qos_mode,
                          int32_t is_dest_only,
                          int32_t has_buffer)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][ALLOC][PORT] step=" << step
              << " now=" << now
              << " port_id=" << port_id
              << " port_bw=" << port_bw
              << " num_tags=" << num_tags
              << " num_live=" << num_live
              << " sum_in=" << sum_in
              << " sum_out=" << sum_out
              << " qos_mode=" << qos_mode
              << " is_dest_only=" << is_dest_only
              << " has_buffer=" << has_buffer
              << "\n";
}

void printSystemAllocSummary(uint64_t step, Time now,
                             int32_t dirty_port_count,
                             int32_t processed_port_count,
                             int32_t dirty_tag_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][ALLOC][SUMMARY] step=" << step
              << " now=" << now
              << " dirty_port_count=" << dirty_port_count
              << " processed_port_count=" << processed_port_count
              << " dirty_tag_count=" << dirty_tag_count
              << "\n";
}

void printSystemEmitSummary(uint64_t step, Time now,
                            int32_t dirty_port_count,
                            int32_t arrival_emit_count,
                            int32_t bwupdate_emit_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][EMIT][SUMMARY] step=" << step
              << " now=" << now
              << " dirty_port_count=" << dirty_port_count
              << " arrival_emit_count=" << arrival_emit_count
              << " bwupdate_emit_count=" << bwupdate_emit_count
              << "\n";
}

void printSystemPfcDetectSummary(uint64_t step, Time now,
                                 int32_t checked_port_count,
                                 int32_t emitted_pfc_count,
                                 int32_t pause_timer_count,
                                 int32_t resume_timer_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][PFCDETECT][SUMMARY] step=" << step
              << " now=" << now
              << " checked_port_count=" << checked_port_count
              << " emitted_pfc_count=" << emitted_pfc_count
              << " pause_timer_count=" << pause_timer_count
              << " resume_timer_count=" << resume_timer_count
              << "\n";
}

void printSystemClearSummary(uint64_t step, Time now,
                             int32_t cleared_port_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][CLEAR][SUMMARY] step=" << step
              << " now=" << now
              << " cleared_port_count=" << cleared_port_count
              << "\n";
}

void printSystemDTSummary(uint64_t step, Time now,
                          double delayed_gap,
                          double pending_gap,
                          double finish_gap,
                          double drain_gap,
                          double backlog_gap,
                          double pfc_pause_gap,
                          double pfc_resume_gap,
                          double chosen_dt)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][DT][SUMMARY] step=" << step
              << " now=" << now
              << " delayed_gap=" << delayed_gap
              << " pending_gap=" << pending_gap
              << " finish_gap=" << finish_gap
              << " drain_gap=" << drain_gap
              << " backlog_gap=" << backlog_gap
              << " pfc_pause_gap=" << pfc_pause_gap
              << " pfc_resume_gap=" << pfc_resume_gap
              << " chosen_dt=" << chosen_dt
              << "\n";
}

void printSystemBufferSummary(uint64_t step, Time now,
                              int32_t processed_port_count,
                              int32_t destroy_count,
                              double total_buf_cnt)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BUFFER][SUMMARY] step=" << step
              << " now=" << now
              << " processed_port_count=" << processed_port_count
              << " destroy_count=" << destroy_count
              << " total_buf_cnt=" << total_buf_cnt
              << "\n";
}
#endif

}
