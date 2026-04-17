#include "sim.hpp"
#include "sim_debug.hpp"

#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::resetNetworkState()
{
    now = 0.0;
    nextPortID = 0;
    numTopoNodes = 0;
    numTopoLinks = 0;
    numPorts = 0;
    numFlowDefs = 0;
    numPendingFlows = 0;
    numFlowRoutes = 0;
    numDelayedEvents = 0;
    numInboxArrival = 0;
    numInboxBwUpdate = 0;
    numInboxPfc = 0;
    numTagIndexEntries = 0;
    numSourceTags = 0;
    numIngressTags = 0;
    numFlowCompletions = 0;
    enableBuffer = 1;
    enablePfc = 0;
    pfcEgress = 0;
    defaultLinkDelay = 0.001;
    propagationInterval = 0.0;
    pfcXoffThreshold = 1e9;
    pfcXonThreshold = 0.5e9;
    dtMin = 0.0;
    qosMode = 0;
    cachedNextDrainTime = std::numeric_limits<Time>::max();
    cachedDrainPortID = -1;
    cachedNextFinishTime = std::numeric_limits<Time>::max();
    numBacklogDrainTimers = 0;
    numPfcPauseTimers = 0;
    numPfcResumeTimers = 0;
    numLastDirtyPortIDs = 0;
    nextDT = 0.0;
    systemLogStep = 0;

    for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
        priorWeights[i] = 0.0;
    }

    for (int32_t i = 0; i < MAX_TOPO_PORTS; i++) {
        portEntities[i] = Entity::none();
        portToNode[i] = -1;
        peerPort[i] = -1;
        backlogDrainPortIDs[i] = -1;
        backlogDrainTimers[i] = 0.0;
        pfcPausePortIDs[i] = -1;
        pfcPauseTimers[i] = 0.0;
        pfcResumePortIDs[i] = -1;
        pfcResumeTimers[i] = 0.0;
        lastDirtyPortIDs[i] = -1;
    }

    for (int32_t i = 0; i < MAX_TOPO_NODES; i++) {
        topoNodes[i] = TopoNodeState {};
        for (int32_t j = 0; j < MAX_TOPO_NODES; j++) {
            linkDelays[i][j] = -1.0;
            routeTable[i][j] = -1;
            ecmpCount[i][j] = 0;
            for (int32_t k = 0; k < MAX_ECMP_NEXT_HOPS; k++) {
                ecmpNextHops[i][j][k] = -1;
            }
        }
    }

    for (int32_t i = 0; i < MAX_TOPO_LINKS; i++) {
        topoLinks[i] = TopoLinkState {};
    }

    for (int32_t i = 0; i < MAX_FLOWS; i++) {
        flowDefs[i] = FlowDef {};
        pendingFlows[i] = FlowDef {};
        flowRoutes[i] = FlowRouteState {};
    }

    for (int32_t i = 0; i < MAX_DELAYED_EVENTS; i++) {
        delayedEvents[i] = DelayedEvent {};
    }
    for (int32_t i = 0; i < MAX_EVENTS_PER_STEP; i++) {
        inboxArrival[i] = FlowArrivalEv {};
        inboxBwUpdate[i] = BwUpdateEv {};
        inboxPfc[i] = PfcControlEv {};
    }
    for (int32_t i = 0; i < MAX_TAG_INDEX; i++) {
        tagIndex[i] = TagIndexEntry {};
    }
    for (int32_t i = 0; i < MAX_SOURCE_TAGS; i++) {
        sourceTags[i] = SourceTagEntry {};
    }
    for (int32_t i = 0; i < MAX_INGRESS_TAGS; i++) {
        ingressTags[i] = IngressTagEntry {};
    }
    for (int32_t i = 0; i < MAX_FLOW_COMPLETIONS; i++) {
        flowCompletions[i] = FlowCompletionEntry {};
    }
}

int32_t Sim::createPort(Engine &ctx, NodeId node_id, int32_t port_idx, Bw port_bw)
{
    Entity port_entity = ctx.makeEntity<Port>();

    int32_t port_id = nextPortID++;
    portEntities[port_id] = port_entity;
    portToNode[port_id] = node_id;
    numPorts = nextPortID;

    ctx.get<DirtyPort>(port_entity).isDirty = 0;
    ctx.get<PortState>(port_entity) = PortState {
        .port_id = port_id,
        .node_id = node_id,
        .port_idx = port_idx,
        .port_bw = port_bw,
        .connected = 0,
        .next_port_id = -1,
    };
    ctx.get<PortBuffer>(port_entity) = PortBuffer {};
    PortPfcConfig pfc_cfg {};
    if (enablePfc != 0) {
        pfc_cfg.pfc_enabled = 1;
        for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
            pfc_cfg.xoff[i] = pfcXoffThreshold;
            pfc_cfg.xon[i] = pfcXonThreshold;
        }
    }
    ctx.get<PortPfcConfig>(port_entity) = pfc_cfg;
    ctx.get<PortPfcState>(port_entity) = PortPfcState {};

    return port_id;
}

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
    for (int32_t i = 0; i < numTopoNodes; i++) {
        topoNodes[i].id = node_defs[i].id;
        topoNodes[i].type = node_defs[i].type;
        topoNodes[i].port_bw = node_defs[i].port_bw;
        topoNodes[i].num_neighbors = 0;
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

        int32_t src_slot = findNodeSlot(input_links[i].src);
        int32_t dst_slot = findNodeSlot(input_links[i].dst);
        if (src_slot >= 0 && dst_slot >= 0) {
            linkDelays[src_slot][dst_slot] = input_links[i].delay;
            linkDelays[dst_slot][src_slot] = input_links[i].delay;
        }
    }

    for (int32_t i = 0; i < numTopoLinks; i++) {
        int32_t src_slot = findNodeSlot(topoLinks[i].src);
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
        topoNodes[src_slot].neighbors[neighbor_idx].port_id = port_id;
        topoNodes[src_slot].neighbors[neighbor_idx].port_entity = portEntities[port_id];
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

    if (init_log_print_enabled) {
        printInitTopoLog(*this, ctx);
    }
}

void Sim::loadFlow(Engine &ctx)
{
    (void)ctx;

    if (network == nullptr) {
        FATAL("Network input is missing");
    }

    int32_t flow_count = network->numFlows;
    if (flow_count > MAX_FLOWS) {
        FATAL("Flow count exceeds MAX_FLOWS");
    }

    numFlowDefs = flow_count;
    numPendingFlows = flow_count;

    for (int32_t i = 0; i < flow_count; i++) {
        flowDefs[i] = network->flows[i];
        pendingFlows[i] = network->flows[i];
    }

    for (int32_t i = 0; i < numPendingFlows; i++) {
        for (int32_t j = i + 1; j < numPendingFlows; j++) {
            if (pendingFlows[j].start_time < pendingFlows[i].start_time) {
                FlowDef tmp = pendingFlows[i];
                pendingFlows[i] = pendingFlows[j];
                pendingFlows[j] = tmp;
            }
        }
    }

    if (init_log_print_enabled) {
        printInitFlowLog(*this);
    }
}

}
