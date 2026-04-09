#include "sim_debug.hpp"

#include <cstdlib>
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
#endif

}

#if defined(__CUDA_ARCH__)
const bool init_log_print_enabled = false;

void printInitTopoLog(const Sim &, Engine &) {}
void printInitFlowLog(const Sim &) {}
#else
const bool init_log_print_enabled = []() {
    const char *env = std::getenv("init_log_print_enabled");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }

    return !(env[0] == '0' && env[1] == '\0');
}();

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

    constexpr int32_t route_samples[][2] = {
        {0, 1},
        {0, 32},
        {32, 0},
        {64, 0},
        {64, 32},
        {0, 64},
        {0, 66},
        {66, 0},
        {64, 66},
    };

    for (const auto &sample : route_samples) {
        int32_t src_slot = sim.findNodeSlot(sample[0]);
        int32_t dst_slot = sim.findNodeSlot(sample[1]);
        NodeId route = -1;
        int32_t ecmp_count = 0;
        if (src_slot >= 0 && dst_slot >= 0) {
            route = sim.routeTable[src_slot][dst_slot];
            ecmp_count = sim.ecmpCount[src_slot][dst_slot];
        }

        std::cout << "[INIT][ROUTE] src=" << sample[0]
                  << " dst=" << sample[1]
                  << " route=" << route
                  << " ecmp_count=" << ecmp_count
                  << " ecmp=";
        if (src_slot >= 0 && dst_slot >= 0) {
            printNodeList(sim.ecmpNextHops[src_slot][dst_slot], ecmp_count);
        } else {
            std::cout << "[]";
        }
        std::cout << "\n";
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
#endif

}
