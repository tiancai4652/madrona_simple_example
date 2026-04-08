#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

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

inline void tick(Engine &,
                 Action &action,
                 Reset &reset,
                 GridPos &,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step)
{
    action = Action::None;
    if (reset.resetNow != 0) {
        reset.resetNow = 0;
    }
    reward.r = 0.f;
    done.episodeDone = 0.f;
    episode_step.step += 1;
}

}

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();

    registry.registerComponent<DirtyPort>();
    registry.registerComponent<PortState>();
    registry.registerComponent<FlowTagState>();
    registry.registerComponent<PortBuffer>();
    registry.registerComponent<PortPfcConfig>();
    registry.registerComponent<PortPfcState>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<Port>();
    registry.registerArchetype<FlowTag>();

    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);
    builder.addToGraph<ParallelForNode<Engine, tick,
        Action, Reset, GridPos, Reward, Done, CurStep>>({});
}

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

    for (int32_t i = 0; i < MAX_TOPO_PORTS; i++) {
        portEntities[i] = Entity::none();
        portToNode[i] = -1;
        peerPort[i] = -1;
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
    ctx.get<PortPfcConfig>(port_entity) = PortPfcConfig {};
    ctx.get<PortPfcState>(port_entity) = PortPfcState {};

    return port_id;
}

void Sim::buildHardcodedTopo(NodeDef *nodes,
                             int32_t &num_nodes,
                             LinkDef *links,
                             int32_t &num_links) const
{
    num_nodes = 4;
    nodes[0] = NodeDef { .id = 0, .type = NodeType::Host,   .port_bw = 100.0 };
    nodes[1] = NodeDef { .id = 1, .type = NodeType::Switch, .port_bw = 100.0 };
    nodes[2] = NodeDef { .id = 2, .type = NodeType::Switch, .port_bw = 100.0 };
    nodes[3] = NodeDef { .id = 3, .type = NodeType::Host,   .port_bw = 100.0 };

    num_links = 4;
    links[0] = LinkDef { .src = 0, .dst = 1, .delay = 0.01, .bandwidth = 100.0 };
    links[1] = LinkDef { .src = 0, .dst = 2, .delay = 0.01, .bandwidth = 100.0 };
    links[2] = LinkDef { .src = 1, .dst = 3, .delay = 0.01, .bandwidth = 100.0 };
    links[3] = LinkDef { .src = 2, .dst = 3, .delay = 0.01, .bandwidth = 100.0 };
}

void Sim::buildHardcodedFlows(FlowDef *flows, int32_t &num_flows) const
{
    num_flows = 2;
    flows[0] = FlowDef {
        .id = 1,
        .src_node = 0,
        .dst_node = 3,
        .size = 1200.0,
        .start_time = 0.0,
        .priority = 0,
    };
    flows[1] = FlowDef {
        .id = 2,
        .src_node = 0,
        .dst_node = 3,
        .size = 2400.0,
        .start_time = 0.0,
        .priority = 1,
    };
}

void Sim::computeRoutes()
{
    int32_t adj_count[MAX_TOPO_NODES] {};
    NodeId adj[MAX_TOPO_NODES][MAX_TOPO_NODES] {};

    for (int32_t i = 0; i < numTopoLinks; i++) {
        int32_t src_slot = findNodeSlot(topoLinks[i].src);
        if (src_slot < 0) {
            continue;
        }

        int32_t idx = adj_count[src_slot]++;
        adj[src_slot][idx] = topoLinks[i].dst;
    }

    int32_t dist[MAX_TOPO_NODES][MAX_TOPO_NODES];
    for (int32_t i = 0; i < MAX_TOPO_NODES; i++) {
        for (int32_t j = 0; j < MAX_TOPO_NODES; j++) {
            dist[i][j] = -1;
            routeTable[i][j] = -1;
            ecmpCount[i][j] = 0;
            for (int32_t k = 0; k < MAX_ECMP_NEXT_HOPS; k++) {
                ecmpNextHops[i][j][k] = -1;
            }
        }
    }

    for (int32_t src_slot = 0; src_slot < numTopoNodes; src_slot++) {
        int32_t queue[MAX_TOPO_NODES];
        int32_t qhead = 0;
        int32_t qtail = 0;
        queue[qtail++] = src_slot;
        dist[src_slot][src_slot] = 0;

        while (qhead < qtail) {
            int32_t u_slot = queue[qhead++];
            for (int32_t i = 0; i < adj_count[u_slot]; i++) {
                NodeId v_id = adj[u_slot][i];
                int32_t v_slot = findNodeSlot(v_id);
                if (v_slot < 0) {
                    continue;
                }
                if (dist[src_slot][v_slot] == -1) {
                    dist[src_slot][v_slot] = dist[src_slot][u_slot] + 1;
                    queue[qtail++] = v_slot;
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

            int32_t shortest_dist = dist[src_slot][dst_slot];
            if (shortest_dist < 0) {
                continue;
            }

            int32_t count = 0;
            for (int32_t i = 0; i < adj_count[src_slot]; i++) {
                NodeId neighbor_id = adj[src_slot][i];
                int32_t neighbor_slot = findNodeSlot(neighbor_id);
                if (neighbor_slot < 0) {
                    continue;
                }

                if (dist[neighbor_slot][dst_slot] >= 0 &&
                    1 + dist[neighbor_slot][dst_slot] == shortest_dist) {
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

void Sim::injectFlowDef(Engine &ctx, const FlowDef &flow)
{
    (void)ctx;

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
        port_path[port_path_len++] = topoNodes[node_slot].neighbors[neighbor_idx].port_id;
    }
    port_path[port_path_len++] = -1;

    if (numFlowRoutes < MAX_FLOWS) {
        FlowRouteState &route = flowRoutes[numFlowRoutes++];
        route.flow_id = flow.id;
        route.num_steps = port_path_len - 1;
        for (int32_t i = 0; i + 1 < port_path_len; i++) {
            route.steps[i].port_id = port_path[i];
            route.steps[i].next_port_id = port_path[i + 1];
        }
    }

}

void Sim::schedulePendingFlows(Engine &ctx)
{
    while (numPendingFlows > 0 && pendingFlows[0].start_time <= now + 1e-15) {
        FlowDef flow = pendingFlows[0];
        for (int32_t i = 1; i < numPendingFlows; i++) {
            pendingFlows[i - 1] = pendingFlows[i];
        }
        numPendingFlows -= 1;
        injectFlowDef(ctx, flow);
    }
}

void Sim::loadTopo(Engine &ctx)
{
    NodeDef node_defs[MAX_TOPO_NODES] {};
    LinkDef input_links[MAX_TOPO_LINKS] {};
    int32_t num_node_defs = 0;
    int32_t num_input_links = 0;
    buildHardcodedTopo(node_defs, num_node_defs, input_links, num_input_links);

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
}

void Sim::loadFlow(Engine &ctx)
{
    (void)ctx;

    FlowDef flows[MAX_FLOWS] {};
    int32_t flow_count = 0;
    buildHardcodedFlows(flows, flow_count);

    numFlowDefs = flow_count;
    numPendingFlows = flow_count;

    for (int32_t i = 0; i < flow_count; i++) {
        flowDefs[i] = flows[i];
        pendingFlows[i] = flows[i];
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

}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength),
      now(0.0),
      nextPortID(0),
      numTopoNodes(0),
      numTopoLinks(0),
      numPorts(0),
      numFlowDefs(0),
      numPendingFlows(0),
      numFlowRoutes(0)
{
    resetNetworkState();
    loadTopo(ctx);
    loadFlow(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
