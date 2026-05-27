#include "sim.hpp"
#include "sim_debug.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {
void Sim::loadTopo(Engine &ctx)
{
    if (network == nullptr) {
        FATAL("Network input is missing");
    }

    const NodeDef *node_defs = network->nodes;
    const LinkDef *input_links = network->links;
    int32_t num_node_defs = network->numNodes;
    int32_t num_input_links = network->numLinks;

    if (num_node_defs > MAX_TOPO_NODES) {
        FATAL("Topology node count exceeds MAX_TOPO_NODES");
    }

    if (num_input_links > MAX_TOPO_LINKS) {
        FATAL("Topology link count exceeds MAX_TOPO_LINKS");
    }

    if (num_input_links * 2 > MAX_TOPO_LINKS) {
        FATAL("Directional topology links exceed MAX_TOPO_LINKS");
    }

    numTopoNodes = num_node_defs;
    if (numTopoNodes > 0) {
        NodeId min_node_id = node_defs[0].id;
        NodeId max_node_id = node_defs[0].id;

        for (int32_t i = 0; i < numTopoNodes; i++) {
            topoNodes[i] = TopoNodeState {};
            topoNodes[i].id = node_defs[i].id;
            topoNodes[i].type = node_defs[i].type;
            topoNodes[i].port_bw = node_defs[i].port_bw;

            if (node_defs[i].id < min_node_id) {
                min_node_id = node_defs[i].id;
            }
            if (node_defs[i].id > max_node_id) {
                max_node_id = node_defs[i].id;
            }
        }

        int64_t lookup_span =
            (int64_t)max_node_id - (int64_t)min_node_id + 1;
        if (lookup_span > 0 && lookup_span <= MAX_TOPO_NODES) {
            nodeLookupBase = min_node_id;
            nodeLookupSpan = (int32_t)lookup_span;
            for (int32_t i = 0; i < nodeLookupSpan; i++) {
                nodeSlotLookup[i] = -1;
            }
            for (int32_t i = 0; i < numTopoNodes; i++) {
                nodeSlotLookup[topoNodes[i].id - nodeLookupBase] = i;
            }
        } else {
            nodeLookupBase = 0;
            nodeLookupSpan = 0;
        }
    }

    numTopoLinks = 0;
    for (int32_t i = 0; i < num_input_links; i++) {
        topoLinks[numTopoLinks++] = TopoLinkState {
            .src = input_links[i].src,
            .dst = input_links[i].dst,
            .delay = input_links[i].delay,
            .bandwidth = input_links[i].bandwidth,
        };
        topoLinks[numTopoLinks++] = TopoLinkState {
            .src = input_links[i].dst,
            .dst = input_links[i].src,
            .delay = input_links[i].delay,
            .bandwidth = input_links[i].bandwidth,
        };

    }

    for (int32_t i = 0; i < numTopoLinks; i++) {
        int32_t src_slot = findNodeSlot(topoLinks[i].src);
        int32_t dst_slot = findNodeSlot(topoLinks[i].dst);
        if (src_slot < 0) {
            continue;
        }

        if (findNeighborSlot(src_slot, topoLinks[i].dst) >= 0) {
            continue;
        }

        int32_t neighbor_idx = topoNodes[src_slot].num_neighbors;
        if (neighbor_idx >= MAX_NODE_NEIGHBORS) {
            FATAL("Topology node exceeds MAX_NODE_NEIGHBORS");
        }
        topoNodes[src_slot].num_neighbors += 1;
        int32_t port_idx = neighbor_idx;
        Bw port_bw = topoLinks[i].bandwidth > 0.0 ? topoLinks[i].bandwidth : topoNodes[src_slot].port_bw;
        int32_t port_id = createPort(ctx, topoLinks[i].src, port_idx, port_bw);
        topoNodes[src_slot].neighbors[neighbor_idx].neighbor_id = topoLinks[i].dst;
        topoNodes[src_slot].neighbors[neighbor_idx].neighbor_slot = dst_slot;
        topoNodes[src_slot].neighbors[neighbor_idx].port_id = port_id;
        topoNodes[src_slot].neighbors[neighbor_idx].port_entity = portEntities[port_id];
        topoNodes[src_slot].neighbors[neighbor_idx].delay = topoLinks[i].delay;
    }

    for (int32_t i = 0; i < numTopoLinks; i++) {
        int32_t src_slot = findNodeSlot(topoLinks[i].src);
        int32_t dst_slot = findNodeSlot(topoLinks[i].dst);
        if (src_slot < 0 || dst_slot < 0) {
            continue;
        }

        int32_t src_neighbor_idx = findNeighborSlot(src_slot, topoLinks[i].dst);
        int32_t dst_neighbor_idx = findNeighborSlot(dst_slot, topoLinks[i].src);
        if (src_neighbor_idx < 0 || dst_neighbor_idx < 0) {
            continue;
        }

        int32_t src_port_id = topoNodes[src_slot].neighbors[src_neighbor_idx].port_id;
        int32_t dst_port_id = topoNodes[dst_slot].neighbors[dst_neighbor_idx].port_id;
        peerPort[dst_port_id] = src_port_id;

        Entity src_entity = portEntities[src_port_id];
        PortState &src_state = ctx.get<PortState>(src_entity);
        src_state.connected = 1;
    }

    computeRoutes();

    if constexpr (init_log_compiled_in) {
        if (init_log_print_enabled) {
            printInitTopoLog(*this, ctx);
        }
    }
}

}
