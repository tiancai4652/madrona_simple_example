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

struct StepScheduleNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().schedulePendingFlows();
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepScheduleNode>(deps);
    }
};

struct StepDeliverNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().deliverEvents();
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepDeliverNode>(deps);
    }
};

struct StepArrivalNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().flowArrivalSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepArrivalNode>(deps);
    }
};

struct StepBwUpdateNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().bwUpdateIngressSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepBwUpdateNode>(deps);
    }
};

struct StepPfcPropagateNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().pfcPropagateSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepPfcPropagateNode>(deps);
    }
};

struct StepPortAllocNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().portBandwidthAllocSystem(ctx, 0.0);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepPortAllocNode>(deps);
    }
};

struct StepPfcDetectNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().pfcThresholdDetectSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepPfcDetectNode>(deps);
    }
};

struct StepDownstreamEmitNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().downstreamEmitSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepDownstreamEmitNode>(deps);
    }
};

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
    TaskGraphNodeID n0 = builder.addToGraph<StepScheduleNode>({});
    TaskGraphNodeID n1 = builder.addToGraph<StepDeliverNode>({n0});
    TaskGraphNodeID n2 = builder.addToGraph<StepArrivalNode>({n1});
    TaskGraphNodeID n3 = builder.addToGraph<StepBwUpdateNode>({n2});
    TaskGraphNodeID n4 = builder.addToGraph<StepPfcPropagateNode>({n3});
    TaskGraphNodeID n5 = builder.addToGraph<StepPortAllocNode>({n4});
    TaskGraphNodeID n6 = builder.addToGraph<StepPfcDetectNode>({n5});
    builder.addToGraph<StepDownstreamEmitNode>({n6});
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

int32_t Sim::findSourceTagIndex(FlowId flow_id) const
{
    for (int32_t i = 0; i < numSourceTags; i++) {
        if (sourceTags[i].flow_id == flow_id) {
            return i;
        }
    }
    return -1;
}

int32_t Sim::findIngressTagIndex(int32_t ingress_port_id, FlowId flow_id) const
{
    for (int32_t i = 0; i < numIngressTags; i++) {
        if (ingressTags[i].ingress_port_id == ingress_port_id &&
            ingressTags[i].flow_id == flow_id) {
            return i;
        }
    }
    return -1;
}

int32_t Sim::findBacklogDrainTimerIndex(int32_t port_id) const
{
    for (int32_t i = 0; i < numBacklogDrainTimers; i++) {
        if (backlogDrainPortIDs[i] == port_id) {
            return i;
        }
    }
    return -1;
}

int32_t Sim::findPfcPauseTimerIndex(int32_t ingress_port_id) const
{
    for (int32_t i = 0; i < numPfcPauseTimers; i++) {
        if (pfcPausePortIDs[i] == ingress_port_id) {
            return i;
        }
    }
    return -1;
}

int32_t Sim::findPfcResumeTimerIndex(int32_t ingress_port_id) const
{
    for (int32_t i = 0; i < numPfcResumeTimers; i++) {
        if (pfcResumePortIDs[i] == ingress_port_id) {
            return i;
        }
    }
    return -1;
}

void Sim::setBacklogDrainTimer(int32_t port_id, Time t)
{
    int32_t idx = findBacklogDrainTimerIndex(port_id);
    if (idx >= 0) {
        backlogDrainTimers[idx] = t;
        return;
    }
    if (numBacklogDrainTimers < MAX_TOPO_PORTS) {
        backlogDrainPortIDs[numBacklogDrainTimers] = port_id;
        backlogDrainTimers[numBacklogDrainTimers] = t;
        numBacklogDrainTimers += 1;
    }
}

void Sim::setPfcPauseTimer(int32_t ingress_port_id, Time t)
{
    int32_t idx = findPfcPauseTimerIndex(ingress_port_id);
    if (idx >= 0) {
        pfcPauseTimers[idx] = t;
        return;
    }
    if (numPfcPauseTimers < MAX_TOPO_PORTS) {
        pfcPausePortIDs[numPfcPauseTimers] = ingress_port_id;
        pfcPauseTimers[numPfcPauseTimers] = t;
        numPfcPauseTimers += 1;
    }
}

void Sim::setPfcResumeTimer(int32_t ingress_port_id, Time t)
{
    int32_t idx = findPfcResumeTimerIndex(ingress_port_id);
    if (idx >= 0) {
        pfcResumeTimers[idx] = t;
        return;
    }
    if (numPfcResumeTimers < MAX_TOPO_PORTS) {
        pfcResumePortIDs[numPfcResumeTimers] = ingress_port_id;
        pfcResumeTimers[numPfcResumeTimers] = t;
        numPfcResumeTimers += 1;
    }
}

void Sim::clearBacklogDrainTimer(int32_t port_id)
{
    int32_t idx = findBacklogDrainTimerIndex(port_id);
    if (idx < 0) {
        return;
    }
    for (int32_t i = idx + 1; i < numBacklogDrainTimers; i++) {
        backlogDrainPortIDs[i - 1] = backlogDrainPortIDs[i];
        backlogDrainTimers[i - 1] = backlogDrainTimers[i];
    }
    numBacklogDrainTimers -= 1;
}

void Sim::clearPfcPauseTimer(int32_t ingress_port_id)
{
    int32_t idx = findPfcPauseTimerIndex(ingress_port_id);
    if (idx < 0) {
        return;
    }
    for (int32_t i = idx + 1; i < numPfcPauseTimers; i++) {
        pfcPausePortIDs[i - 1] = pfcPausePortIDs[i];
        pfcPauseTimers[i - 1] = pfcPauseTimers[i];
    }
    numPfcPauseTimers -= 1;
}

void Sim::clearPfcResumeTimer(int32_t ingress_port_id)
{
    int32_t idx = findPfcResumeTimerIndex(ingress_port_id);
    if (idx < 0) {
        return;
    }
    for (int32_t i = idx + 1; i < numPfcResumeTimers; i++) {
        pfcResumePortIDs[i - 1] = pfcResumePortIDs[i];
        pfcResumeTimers[i - 1] = pfcResumeTimers[i];
    }
    numPfcResumeTimers -= 1;
}

Entity Sim::findTag(int32_t port_id, FlowId flow_id) const
{
    for (int32_t i = 0; i < numTagIndexEntries; i++) {
        if (tagIndex[i].port_id == port_id && tagIndex[i].flow_id == flow_id) {
            return tagIndex[i].entity;
        }
    }

    return Entity::none();
}

void Sim::materializeBacklog(FlowTagState &tag, Time at_time)
{
    if (tag.is_source != 0) {
        tag.backlog = 0.0;
        tag.last_backlog_time = at_time;
        return;
    }

    if (at_time > tag.last_backlog_time) {
        Time elapsed = at_time - tag.last_backlog_time;
        tag.backlog += (tag.in_bw - tag.out_bw) * elapsed;
        if (tag.backlog < 0.0) {
            tag.backlog = 0.0;
        }
        tag.last_backlog_time = at_time;
    }
}

void Sim::recordFlowCompletion(FlowId flow_id, Time end_time)
{
    for (int32_t i = 0; i < numFlowCompletions; i++) {
        if (flowCompletions[i].flow_id == flow_id) {
            if (end_time > flowCompletions[i].record.end_time) {
                flowCompletions[i].record.end_time = end_time;
            }
            return;
        }
    }

    if (numFlowCompletions >= MAX_FLOW_COMPLETIONS) {
        return;
    }

    for (int32_t i = 0; i < numFlowDefs; i++) {
        if (flowDefs[i].id == flow_id) {
            FlowCompletionEntry &entry = flowCompletions[numFlowCompletions++];
            entry.flow_id = flow_id;
            entry.record = FlowCompletionRecord {
                .flow_id = flow_id,
                .src_node = flowDefs[i].src_node,
                .dst_node = flowDefs[i].dst_node,
                .size = flowDefs[i].size,
                .start_time = flowDefs[i].start_time,
                .end_time = end_time,
                .priority = flowDefs[i].priority,
            };
            return;
        }
    }
}

void Sim::destroyTag(Engine &ctx,
                     Entity tag_entity,
                     bool propagate_cleanup,
                     Time logical_now)
{
    if (tag_entity == Entity::none()) {
        return;
    }

    FlowTagState tag = ctx.get<FlowTagState>(tag_entity);
    Time effective_now = logical_now >= 0.0 ? logical_now : now;

    if (tag.next_port_id < 0) {
        recordFlowCompletion(tag.flow_id, effective_now);
    }

    if (propagate_cleanup && tag.next_port_id >= 0) {
        if (tag.downstream_created != 0) {
            int32_t src_node_slot = findNodeSlot(portToNode[tag.port_id]);
            int32_t dst_node_slot = findNodeSlot(portToNode[tag.next_port_id]);
            Time delay = 0.0;
            if (src_node_slot >= 0 && dst_node_slot >= 0) {
                delay = linkDelays[src_node_slot][dst_node_slot];
                if (delay < 0.0) {
                    delay = 0.0;
                }
            }
            DelayedEvent ev {};
            ev.t = effective_now + delay;
            ev.type = DelayedEvent::Type::BwUpdate;
            ev.bwupd = BwUpdateEv {
                .port_id = tag.next_port_id,
                .flow_id = tag.flow_id,
                .in_bw = 0.0,
            };
            pushDelayedEvent(ev);
        } else {
            int32_t cur = tag.port_id;
            int32_t nxt = lookupFlowRouteNext(tag.flow_id, cur);
            while (nxt >= 0) {
                cur = nxt;
                nxt = lookupFlowRouteNext(tag.flow_id, cur);
            }
            if (cur != tag.port_id) {
                recordFlowCompletion(tag.flow_id, effective_now);
            }
        }
    }

    for (int32_t i = 0; i < numTagIndexEntries; i++) {
        if (tagIndex[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numTagIndexEntries; j++) {
                tagIndex[j - 1] = tagIndex[j];
            }
            numTagIndexEntries -= 1;
            break;
        }
    }

    for (int32_t i = 0; i < numIngressTags; i++) {
        if (ingressTags[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numIngressTags; j++) {
                ingressTags[j - 1] = ingressTags[j];
            }
            numIngressTags -= 1;
            i -= 1;
        }
    }

    for (int32_t i = 0; i < numSourceTags; i++) {
        if (sourceTags[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numSourceTags; j++) {
                sourceTags[j - 1] = sourceTags[j];
            }
            numSourceTags -= 1;
            break;
        }
    }

    if (tag.port_id >= 0 && tag.port_id < numPorts) {
        Entity port_entity = portEntities[tag.port_id];
        if (port_entity != Entity::none()) {
            ctx.get<DirtyPort>(port_entity).isDirty = 1;
        }
    }

    ctx.destroyEntity(tag_entity);
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

void Sim::materializeRemaining(FlowTagState &tag, Time at_time)
{
    if (tag.is_source != 0 && at_time > tag.last_remaining_time) {
        Time elapsed = at_time - tag.last_remaining_time;
        tag.remaining -= tag.out_bw * elapsed;
        if (tag.remaining < 0.0) {
            tag.remaining = 0.0;
        }
        tag.last_remaining_time = at_time;
    }
}

double Sim::drainBufferChunks(PriorityBuffer &pb, double drain_bytes)
{
    if (drain_bytes <= 1e-15 || pb.num_chunks <= 0) {
        return 0.0;
    }

    double remaining = drain_bytes;
    double total = 0.0;

    while (remaining > 1e-15 && pb.num_chunks > 0) {
        BufferChunk &front = pb.buf_chunks[pb.head];
        if (front.chunk_bytes <= remaining + 1e-15) {
            remaining -= front.chunk_bytes;
            total += front.chunk_bytes;
            front = BufferChunk {};
            pb.head = (pb.head + 1) % MAX_BUFFER_CHUNKS;
            pb.num_chunks -= 1;
        } else {
            front.chunk_bytes -= remaining;
            total += remaining;
            remaining = 0.0;
        }
    }

    if (pb.num_chunks <= 0) {
        pb.head = 0;
        pb.tail = 0;
    }

    return total;
}

void Sim::alignChunksWithBufCnt(PriorityBuffer &pb)
{
    if (pb.num_chunks <= 0 || pb.buf_cnt < 1e-15) {
        return;
    }

    double chunk_sum = 0.0;
    for (int32_t i = 0; i < pb.num_chunks; i++) {
        int32_t idx = (pb.head + i) % MAX_BUFFER_CHUNKS;
        chunk_sum += pb.buf_chunks[idx].chunk_bytes;
    }

    double diff = pb.buf_cnt - chunk_sum;
    constexpr double EPS = 1.0;
    if (diff > EPS) {
        if (pb.num_chunks > 0 && pb.num_chunks < MAX_BUFFER_CHUNKS) {
            int32_t back_idx = (pb.tail - 1 + MAX_BUFFER_CHUNKS) % MAX_BUFFER_CHUNKS;
            BufferChunk gc {};
            gc.chunk_bytes = diff;
            gc.num_weights = pb.buf_chunks[back_idx].num_weights;
            for (int32_t i = 0; i < gc.num_weights; i++) {
                gc.weights[i] = pb.buf_chunks[back_idx].weights[i];
            }
            pb.buf_chunks[pb.tail] = gc;
            pb.tail = (pb.tail + 1) % MAX_BUFFER_CHUNKS;
            pb.num_chunks += 1;
        }
    } else if (diff < -EPS) {
        drainBufferChunks(pb, -diff);
        chunk_sum = 0.0;
        for (int32_t i = 0; i < pb.num_chunks; i++) {
            int32_t idx = (pb.head + i) % MAX_BUFFER_CHUNKS;
            chunk_sum += pb.buf_chunks[idx].chunk_bytes;
        }
        pb.buf_cnt = chunk_sum;
    }
}

void Sim::materializeBufCnt(PortBuffer &port_buf, Time at_time)
{
    if (at_time > port_buf.last_update_time + 1e-15) {
        Time elapsed = at_time - port_buf.last_update_time;
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            PriorityBuffer &pb = port_buf.prior_bufs[pri];
            pb.buf_cnt += pb.net_buffer_rate * elapsed;
            if (pb.buf_cnt < 0.0) {
                pb.buf_cnt = 0.0;
            }
        }
        port_buf.last_update_time = at_time;
    }
}

Entity Sim::createTagOnPort(Engine &ctx,
                            int32_t port_id,
                            FlowId flow_id,
                            Bw in_bw,
                            Bytes size,
                            bool is_source,
                            int32_t priority)
{
    if (port_id < 0 || port_id >= numPorts) {
        return Entity::none();
    }

    Entity port_entity = portEntities[port_id];
    if (port_entity == Entity::none()) {
        return Entity::none();
    }

    Entity tag_entity = ctx.makeEntity<FlowTag>();
    FlowTagState tag {};
    tag.port_id = port_id;
    tag.flow_id = flow_id;
    tag.priority = priority;
    tag.in_bw = in_bw;
    tag.out_bw = 0.0;
    tag.prev_out_bw = 0.0;
    tag.backlog = 0.0;
    tag.last_backlog_time = now;
    tag.remaining = size;
    tag.last_remaining_time = now;
    tag.is_source = is_source ? 1 : 0;
    tag.downstream_created = 0;
    tag.next_port_id = lookupFlowRouteNext(flow_id, port_id);
    tag.ingress_port_id = -1;

    if (!is_source) {
        for (int32_t i = 0; i < numFlowRoutes; i++) {
            if (flowRoutes[i].flow_id != flow_id) {
                continue;
            }
            for (int32_t j = 0; j < flowRoutes[i].num_steps; j++) {
                if (flowRoutes[i].steps[j].next_port_id == port_id) {
                    int32_t upstream_port = flowRoutes[i].steps[j].port_id;
                    if (upstream_port >= 0 && upstream_port < numPorts) {
                        tag.ingress_port_id = peerPort[upstream_port];
                    }
                    break;
                }
            }
        }
    }

    ctx.get<FlowTagState>(tag_entity) = tag;

    if (numTagIndexEntries < MAX_TAG_INDEX) {
        tagIndex[numTagIndexEntries++] = TagIndexEntry {
            .port_id = port_id,
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    if (tag.ingress_port_id >= 0 && numIngressTags < MAX_INGRESS_TAGS) {
        ingressTags[numIngressTags++] = IngressTagEntry {
            .ingress_port_id = tag.ingress_port_id,
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    if (is_source && numSourceTags < MAX_SOURCE_TAGS) {
        sourceTags[numSourceTags++] = SourceTagEntry {
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    ctx.get<DirtyPort>(port_entity).isDirty = 1;
    PortBuffer &port_buf = ctx.get<PortBuffer>(port_entity);
    if (port_buf.last_update_time < now) {
        port_buf.last_update_time = now;
    }

    return tag_entity;
}

void Sim::injectFlow(int32_t src_port_id, const FlowDef &flow)
{
    int32_t src_slot = findNodeSlot(flow.src_node);
    Bw src_in_bw = 0.0;
    if (src_slot >= 0) {
        src_in_bw = topoNodes[src_slot].port_bw;
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

void Sim::injectFlowDef(const FlowDef &flow)
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
        port_path[port_path_len++] = topoNodes[node_slot].neighbors[neighbor_idx].port_id;
    }
    port_path[port_path_len++] = -1;

    bool exists = false;
    for (int32_t i = 0; i < numFlowRoutes; i++) {
        if (flowRoutes[i].flow_id == flow.id) {
            exists = true;
            break;
        }
    }

    if (!exists && numFlowRoutes < MAX_FLOWS) {
        FlowRouteState &route = flowRoutes[numFlowRoutes++];
        route.flow_id = flow.id;
        route.num_steps = port_path_len - 1;
        for (int32_t i = 0; i + 1 < port_path_len; i++) {
            route.steps[i].port_id = port_path[i];
            route.steps[i].next_port_id = port_path[i + 1];
        }
    }

    int32_t src_port_id = topoNodes[src_slot].neighbors[first_neighbor_idx].port_id;
    injectFlow(src_port_id, flow);
}

void Sim::schedulePendingFlows()
{
    while (numPendingFlows > 0 && pendingFlows[0].start_time <= now + 1e-15) {
        FlowDef flow = pendingFlows[0];
        for (int32_t i = 1; i < numPendingFlows; i++) {
            pendingFlows[i - 1] = pendingFlows[i];
        }
        numPendingFlows -= 1;
        injectFlowDef(flow);
    }
}

void Sim::deliverEvents()
{
    numInboxArrival = 0;
    numInboxBwUpdate = 0;
    numInboxPfc = 0;

    int32_t write_idx = 0;
    for (int32_t i = 0; i < numDelayedEvents; i++) {
        if (delayedEvents[i].t <= now + 1e-15) {
            if (delayedEvents[i].type == DelayedEvent::Type::Arrival) {
                if (numInboxArrival < MAX_EVENTS_PER_STEP) {
                    inboxArrival[numInboxArrival++] = delayedEvents[i].arrival;
                }
            } else if (delayedEvents[i].type == DelayedEvent::Type::BwUpdate) {
                if (numInboxBwUpdate < MAX_EVENTS_PER_STEP) {
                    inboxBwUpdate[numInboxBwUpdate++] = delayedEvents[i].bwupd;
                }
            } else {
                if (numInboxPfc < MAX_EVENTS_PER_STEP) {
                    inboxPfc[numInboxPfc++] = delayedEvents[i].pfcctrl;
                }
            }
        } else {
            delayedEvents[write_idx++] = delayedEvents[i];
        }
    }
    numDelayedEvents = write_idx;
}

void Sim::flowArrivalSystem(Engine &ctx)
{
    for (int32_t i = 0; i < numInboxArrival; i++) {
        const FlowArrivalEv &ev = inboxArrival[i];
        if (ev.port_id < 0 || ev.port_id >= numPorts) {
            continue;
        }

        Entity port_entity = portEntities[ev.port_id];
        if (port_entity == Entity::none()) {
            continue;
        }

        Entity existing = findTag(ev.port_id, ev.flow_id);
        if (existing != Entity::none()) {
            FlowTagState &tag = ctx.get<FlowTagState>(existing);
            tag.in_bw = ev.in_bw;
            if (ev.is_source != 0) {
                tag.is_source = 1;
                tag.remaining = ev.size;
            }
            ctx.get<DirtyPort>(port_entity).isDirty = 1;
            continue;
        }

        createTagOnPort(ctx, ev.port_id, ev.flow_id, ev.in_bw,
            ev.size, ev.is_source != 0, ev.priority);
    }
}

void Sim::bwUpdateIngressSystem(Engine &ctx)
{
    for (int32_t i = 0; i < numInboxBwUpdate; i++) {
        const BwUpdateEv &ev = inboxBwUpdate[i];
        if (ev.port_id < 0 || ev.port_id >= numPorts) {
            continue;
        }

        Entity port_entity = portEntities[ev.port_id];
        if (port_entity == Entity::none()) {
            continue;
        }

        Entity existing = findTag(ev.port_id, ev.flow_id);

        if (ev.in_bw == 0.0) {
            if (existing != Entity::none()) {
                FlowTagState &tag = ctx.get<FlowTagState>(existing);
                materializeBacklog(tag, now);
                if (enableBuffer != 0 && tag.backlog > 1e-15) {
                    tag.in_bw = 0.0;
                    ctx.get<DirtyPort>(port_entity).isDirty = 1;
                } else {
                    destroyTag(ctx, existing, true, now);
                }
            }
            continue;
        }

        if (existing == Entity::none()) {
            bool has_cleanup = false;
            for (int32_t j = 0; j < numInboxBwUpdate; j++) {
                if (inboxBwUpdate[j].port_id == ev.port_id &&
                    inboxBwUpdate[j].flow_id == ev.flow_id &&
                    inboxBwUpdate[j].in_bw == 0.0) {
                    has_cleanup = true;
                    break;
                }
            }

            if (has_cleanup) {
                int32_t next_port = lookupFlowRouteNext(ev.flow_id, ev.port_id);
                if (next_port >= 0) {
                    DelayedEvent cleanup_ev {};
                    cleanup_ev.t = computePropagationTimeForPort(ev.port_id, next_port);
                    cleanup_ev.type = DelayedEvent::Type::BwUpdate;
                    cleanup_ev.bwupd = BwUpdateEv {
                        .port_id = next_port,
                        .flow_id = ev.flow_id,
                        .in_bw = 0.0,
                    };
                    pushDelayedEvent(cleanup_ev);
                } else {
                    recordFlowCompletion(ev.flow_id, now);
                }
                continue;
            }

            int32_t pri = 0;
            for (int32_t j = 0; j < numFlowDefs; j++) {
                if (flowDefs[j].id == ev.flow_id) {
                    pri = flowDefs[j].priority;
                    break;
                }
            }
            createTagOnPort(ctx, ev.port_id, ev.flow_id, ev.in_bw, 0.0, false, pri);
        } else {
            FlowTagState &tag = ctx.get<FlowTagState>(existing);
            if (tag.in_bw != ev.in_bw) {
                materializeBacklog(tag, now);
                tag.in_bw = ev.in_bw;
            }
            ctx.get<DirtyPort>(port_entity).isDirty = 1;
        }
    }
}

void Sim::pfcPropagateSystem(Engine &ctx)
{
    if (enablePfc == 0 || numInboxPfc == 0) {
        return;
    }

    for (int32_t i = 0; i < numInboxPfc; i++) {
        const PfcControlEv &ev = inboxPfc[i];
        if (ev.target_port_id < 0 || ev.target_port_id >= numPorts) {
            continue;
        }

        Entity target_port = portEntities[ev.target_port_id];
        if (target_port == Entity::none()) {
            continue;
        }

        PortPfcState &state = ctx.get<PortPfcState>(target_port);
        if (ev.priority >= 0 && ev.priority < PFC_MAX_PRIORITY) {
            state.paused[ev.priority] = ev.paused;
        }
        ctx.get<DirtyPort>(target_port).isDirty = 1;
    }
}

void Sim::portBandwidthAllocSystem(Engine &ctx, Time dt)
{
    (void)dt;

    int32_t dirty_ports[MAX_TOPO_PORTS] {};
    int32_t num_dirty_ports = 0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        if (ctx.get<DirtyPort>(port_e).isDirty != 0) {
            dirty_ports[num_dirty_ports++] = port_id;
        }
    }

    if (num_dirty_ports == 0) {
        return;
    }

    for (int32_t i = 0; i < num_dirty_ports; i++) {
        int32_t pid = dirty_ports[i];
        if (pid == cachedDrainPortID) {
            cachedNextDrainTime = std::numeric_limits<Time>::max();
            cachedDrainPortID = -1;
        }
        clearBacklogDrainTimer(pid);
    }

    for (int32_t d = 0; d < num_dirty_ports; d++) {
        int32_t port_id = dirty_ports[d];
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        double port_bw = ctx.get<PortState>(port_e).port_bw;
        PortBuffer *port_buf = &ctx.get<PortBuffer>(port_e);
        PortPfcState *pfc_state = enablePfc ? &ctx.get<PortPfcState>(port_e) : nullptr;

        materializeBufCnt(*port_buf, now);
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            PriorityBuffer &pb = port_buf->prior_bufs[pri];
            if (pb.buf_cnt > 1.0 && pb.num_chunks <= 0) {
                BufferChunk comp {};
                comp.chunk_bytes = pb.buf_cnt;
                int32_t weight_count = 0;
                for (int32_t i = 0; i < numTagIndexEntries; i++) {
                    if (tagIndex[i].port_id != port_id) {
                        continue;
                    }
                    FlowTagState &tag = ctx.get<FlowTagState>(tagIndex[i].entity);
                    int p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                    if (p != pri) {
                        continue;
                    }
                    if (weight_count < MAX_CHUNK_WEIGHTS) {
                        comp.weights[weight_count].flow_id = tag.flow_id;
                        comp.weights[weight_count].weight = 1.0;
                        weight_count += 1;
                    }
                }
                comp.num_weights = weight_count;
                if (weight_count > 0) {
                    for (int32_t i = 0; i < weight_count; i++) {
                        comp.weights[i].weight = 1.0 / (double)weight_count;
                    }
                    pb.buf_chunks[pb.tail] = comp;
                    pb.tail = (pb.tail + 1) % MAX_BUFFER_CHUNKS;
                    pb.num_chunks += 1;
                }
            }
            if (pb.buf_cnt < 1e-15 && pb.num_chunks > 0) {
                double chunk_sum = 0.0;
                for (int32_t i = 0; i < pb.num_chunks; i++) {
                    int32_t idx = (pb.head + i) % MAX_BUFFER_CHUNKS;
                    chunk_sum += pb.buf_chunks[idx].chunk_bytes;
                }
                if (chunk_sum > 1e-15) {
                    pb.buf_cnt = chunk_sum;
                }
            }
            alignChunksWithBufCnt(pb);
        }

        Entity tags[MAX_TAG_INDEX] {};
        int32_t num_tags = 0;
        double sum_in = 0.0;
        for (int32_t i = 0; i < numTagIndexEntries; i++) {
            if (tagIndex[i].port_id != port_id) {
                continue;
            }
            Entity tag_e = tagIndex[i].entity;
            if (tag_e == Entity::none()) {
                continue;
            }
            tags[num_tags++] = tag_e;
            FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            sum_in += tag.in_bw;
        }
        if (num_tags == 0) {
            continue;
        }

        for (int32_t i = 0; i < num_tags; i++) {
            FlowTagState &tag = ctx.get<FlowTagState>(tags[i]);
            if (tag.is_source != 0) {
                materializeRemaining(tag, now);
            }
            materializeBacklog(tag, now);
            tag.prev_out_bw = tag.out_bw;
            tag.out_bw = 0.0;
        }

        for (int32_t i = 0; i < num_tags; i++) {
            FlowTagState &tag = ctx.get<FlowTagState>(tags[i]);
            if (tag.is_source == 0 && tag.in_bw == 0.0 && tag.backlog < 1.0) {
                tag.backlog = 0.0;
                destroyTag(ctx, tags[i], true, now);
                tags[i] = Entity::none();
            }
        }

        Entity live_tags[MAX_TAG_INDEX] {};
        int32_t num_live = 0;
        double live_sum_in = 0.0;
        for (int32_t i = 0; i < num_tags; i++) {
            if (tags[i] == Entity::none()) {
                continue;
            }
            live_tags[num_live++] = tags[i];
            FlowTagState &tag = ctx.get<FlowTagState>(tags[i]);
            live_sum_in += tag.in_bw;
        }
        if (num_live == 0) {
            continue;
        }

        bool is_dest_only = false;
        if (qosMode != QOS_NONE) {
            int32_t node_slot = findNodeSlot(portToNode[port_id]);
            if (node_slot >= 0 && topoNodes[node_slot].type == NodeType::Host) {
                is_dest_only = true;
                for (int32_t i = 0; i < num_live; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                    if (tag.next_port_id >= 0) {
                        is_dest_only = false;
                        break;
                    }
                }
            }
        }

        double out_total = 0.0;
        if ((qosMode == QOS_SP || qosMode == QOS_WRR) && !is_dest_only) {
            Entity pri_tags[PFC_MAX_PRIORITY][MAX_TAG_INDEX] {};
            int32_t pri_counts[PFC_MAX_PRIORITY] {};
            double pri_in_sum[PFC_MAX_PRIORITY] {};
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                pri_tags[pri][pri_counts[pri]++] = live_tags[i];
                pri_in_sum[pri] += tag.in_bw;
            }

            if (qosMode == QOS_SP) {
                double remaining_bw = port_bw;
                for (int32_t pri = 0; pri < PFC_MAX_PRIORITY && remaining_bw > 1e-15; pri++) {
                    if (pri_counts[pri] == 0) {
                        continue;
                    }
                    if (pfc_state && pfc_state->paused[pri] != 0) {
                        continue;
                    }
                    PriorityBuffer &pb = port_buf->prior_bufs[pri];
                    bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                    double allocated = 0.0;
                    if (has_buf) {
                        BufferChunk &oldest = pb.buf_chunks[pb.head];
                        double active_weight_sum = 0.0;
                        for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                            Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                            if (t != Entity::none()) {
                                FlowTagState &tag = ctx.get<FlowTagState>(t);
                                if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                    active_weight_sum += oldest.weights[wi].weight;
                                }
                            }
                        }
                        if (active_weight_sum > 1e-15) {
                            for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                                Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                                if (t != Entity::none()) {
                                    FlowTagState &tag = ctx.get<FlowTagState>(t);
                                    if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                        tag.out_bw = remaining_bw * (oldest.weights[wi].weight / active_weight_sum);
                                        allocated += tag.out_bw;
                                    }
                                }
                            }
                        }
                        if (allocated < 1e-15) {
                            double bl_sum = 0.0;
                            for (int32_t i = 0; i < pri_counts[pri]; i++) {
                                FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                                double d = std::max(tag.in_bw, 0.0);
                                if (tag.backlog > 1e-15) {
                                    d += tag.backlog;
                                }
                                bl_sum += d;
                            }
                            if (bl_sum > 1e-15) {
                                for (int32_t i = 0; i < pri_counts[pri]; i++) {
                                    FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                                    double d = std::max(tag.in_bw, 0.0);
                                    if (tag.backlog > 1e-15) {
                                        d += tag.backlog;
                                    }
                                    tag.out_bw = remaining_bw * (d / bl_sum);
                                    allocated += tag.out_bw;
                                }
                            }
                        }
                    } else {
                        double demand = pri_in_sum[pri];
                        double alloc_bw = std::min(demand, remaining_bw);
                        if (demand > 1e-15) {
                            double scale = alloc_bw / demand;
                            for (int32_t i = 0; i < pri_counts[pri]; i++) {
                                FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                                tag.out_bw = tag.in_bw * scale;
                                allocated += tag.out_bw;
                            }
                        }
                    }
                    remaining_bw -= allocated;
                    if (remaining_bw < 0.0) {
                        remaining_bw = 0.0;
                    }
                }
            } else {
                bool active[PFC_MAX_PRIORITY] {};
                double weight_sum = 0.0;
                bool has_backlog[PFC_MAX_PRIORITY] {};
                double base_share[PFC_MAX_PRIORITY] {};
                double alloc[PFC_MAX_PRIORITY] {};
                double leftover = 0.0;
                for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                    if (pri_counts[pri] == 0) {
                        continue;
                    }
                    if (pfc_state && pfc_state->paused[pri] != 0) {
                        continue;
                    }
                    if (priorWeights[pri] < 1e-15) {
                        continue;
                    }
                    active[pri] = true;
                    weight_sum += priorWeights[pri];
                }
                if (weight_sum > 1e-15) {
                    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                        if (!active[pri]) {
                            continue;
                        }
                        base_share[pri] = port_bw * (priorWeights[pri] / weight_sum);
                        PriorityBuffer &pb = port_buf->prior_bufs[pri];
                        bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                        bool any_backlog = false;
                        for (int32_t i = 0; i < pri_counts[pri]; i++) {
                            if (ctx.get<FlowTagState>(pri_tags[pri][i]).backlog > 1e-15) {
                                any_backlog = true;
                                break;
                            }
                        }
                        has_backlog[pri] = has_buf || any_backlog;
                        if (has_backlog[pri]) {
                            alloc[pri] = base_share[pri];
                        } else {
                            double demand = pri_in_sum[pri];
                            alloc[pri] = std::min(base_share[pri], demand);
                            leftover += base_share[pri] - alloc[pri];
                        }
                    }
                    if (leftover > 1e-15) {
                        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                            if (!active[pri] || has_backlog[pri]) {
                                continue;
                            }
                            double demand = pri_in_sum[pri];
                            if (demand > alloc[pri]) {
                                double top_up = std::min(demand - alloc[pri], leftover);
                                alloc[pri] += top_up;
                                leftover -= top_up;
                                if (leftover < 1e-15) {
                                    break;
                                }
                            }
                        }
                        if (leftover > 1e-15) {
                            double bl_weight_sum = 0.0;
                            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                                if (active[pri] && has_backlog[pri]) {
                                    bl_weight_sum += priorWeights[pri];
                                }
                            }
                            if (bl_weight_sum > 1e-15) {
                                for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                                    if (active[pri] && has_backlog[pri]) {
                                        alloc[pri] += leftover * (priorWeights[pri] / bl_weight_sum);
                                    }
                                }
                            }
                        }
                    }
                    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                        if (!active[pri] || alloc[pri] < 1e-15) {
                            continue;
                        }
                        PriorityBuffer &pb = port_buf->prior_bufs[pri];
                        bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                        if (has_buf) {
                            BufferChunk &oldest = pb.buf_chunks[pb.head];
                            double aw = 0.0;
                            for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                                Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                                if (t != Entity::none()) {
                                    FlowTagState &tag = ctx.get<FlowTagState>(t);
                                    if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                        aw += oldest.weights[wi].weight;
                                    }
                                }
                            }
                            if (aw > 1e-15) {
                                for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                                    Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                                    if (t != Entity::none()) {
                                        FlowTagState &tag = ctx.get<FlowTagState>(t);
                                        if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                            tag.out_bw = alloc[pri] * (oldest.weights[wi].weight / aw);
                                        }
                                    }
                                }
                            }
                        } else {
                            double sum_pri_in = pri_in_sum[pri];
                            if (sum_pri_in > 1e-15) {
                                double scale = std::min(1.0, alloc[pri] / sum_pri_in);
                                for (int32_t i = 0; i < pri_counts[pri]; i++) {
                                    FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                                    tag.out_bw = tag.in_bw * scale;
                                }
                            }
                        }
                    }
                }
            }

            double pri_out_sum[PFC_MAX_PRIORITY] {};
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                out_total += tag.out_bw;
                int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                pri_out_sum[p] += tag.out_bw;
            }
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                port_buf->prior_bufs[pri].net_buffer_rate = 0.0;
            }
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                port_buf->prior_bufs[p].net_buffer_rate += tag.in_bw;
            }
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                port_buf->prior_bufs[pri].net_buffer_rate -= pri_out_sum[pri];
                PriorityBuffer &pb = port_buf->prior_bufs[pri];
                alignChunksWithBufCnt(pb);
                if (enableBuffer && pb.buf_cnt > 1e-15 && pb.num_chunks > 0) {
                    double out_pri = pri_out_sum[pri];
                    if (out_pri > 1e-6) {
                        BufferChunk &front = pb.buf_chunks[pb.head];
                        if (front.chunk_bytes > 1e-15) {
                            Time td = front.chunk_bytes / out_pri;
                            if (td > 1e-15 && td < cachedNextDrainTime) {
                                cachedNextDrainTime = td;
                                cachedDrainPortID = port_id;
                            }
                        }
                    }
                }
            }
        } else if (is_dest_only && qosMode != QOS_NONE) {
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                PriorityBuffer &pb = port_buf->prior_bufs[p];
                bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                bool needs_drain = tag.backlog > 1.0;
                if (has_buf || needs_drain) {
                    tag.out_bw = std::max(tag.in_bw, port_bw / (double)num_live);
                } else {
                    tag.out_bw = tag.in_bw;
                }
                out_total += tag.out_bw;
            }
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                double pi = 0.0;
                double po = 0.0;
                for (int32_t i = 0; i < num_live; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                    int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                    if (p == pri) {
                        pi += tag.in_bw;
                        po += tag.out_bw;
                    }
                }
                port_buf->prior_bufs[pri].net_buffer_rate = pi - po;
            }
        } else {
            PriorityBuffer &pb = port_buf->prior_bufs[0];
            bool has_buffer = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
            if (!has_buffer && live_sum_in < 1e-18) {
                continue;
            }
            bool is_congested = live_sum_in >= port_bw;
            if (has_buffer) {
                BufferChunk &oldest = pb.buf_chunks[pb.head];
                double active_weight_sum = 0.0;
                for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                    Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                    if (t != Entity::none()) {
                        FlowTagState &tag = ctx.get<FlowTagState>(t);
                        if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                            active_weight_sum += oldest.weights[wi].weight;
                        }
                    }
                }
                if (active_weight_sum > 1e-15) {
                    for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                        Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                        if (t != Entity::none()) {
                            FlowTagState &tag = ctx.get<FlowTagState>(t);
                            if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                tag.out_bw = port_bw * (oldest.weights[wi].weight / active_weight_sum);
                            }
                        }
                    }
                } else if (live_sum_in > 1e-15) {
                    for (int32_t i = 0; i < num_live; i++) {
                        FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                        tag.out_bw = port_bw * (tag.in_bw / live_sum_in);
                    }
                }
            } else {
                if (live_sum_in >= 1e-18) {
                    if (is_congested) {
                        for (int32_t i = 0; i < num_live; i++) {
                            FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                            tag.out_bw = tag.in_bw * (port_bw / live_sum_in);
                        }
                    } else {
                        for (int32_t i = 0; i < num_live; i++) {
                            FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                            tag.out_bw = tag.in_bw;
                        }
                    }
                }
            }
            if (pfc_state) {
                for (int32_t i = 0; i < num_live; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                    if (pfc_state->paused[std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1)] != 0) {
                        tag.out_bw = 0.0;
                    }
                }
            }
            for (int32_t i = 0; i < num_live; i++) {
                out_total += ctx.get<FlowTagState>(live_tags[i]).out_bw;
            }
            if (enableBuffer && has_buffer) {
                BufferChunk &front = pb.buf_chunks[pb.head];
                if (port_bw > 1e-6 && front.chunk_bytes > 1e-15) {
                    Time td = front.chunk_bytes / port_bw;
                    if (td > 1e-15 && td < cachedNextDrainTime) {
                        cachedNextDrainTime = td;
                        cachedDrainPortID = port_id;
                    }
                }
            }
            pb.net_buffer_rate = live_sum_in - out_total;
        }

        for (int32_t i = 0; i < num_live; i++) {
            FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
            if (tag.is_source != 0) {
                if (tag.remaining > 0.0 && tag.remaining < 1.0) {
                    tag.remaining = 0.0;
                }
                if (tag.out_bw > 1e-15 && tag.remaining > 0.0) {
                    Time t_finish = tag.remaining / tag.out_bw;
                    if (t_finish > 0.0 && t_finish < cachedNextFinishTime) {
                        cachedNextFinishTime = t_finish;
                    }
                }
            }
            if (tag.is_source == 0 && tag.in_bw == 0.0 && tag.backlog > 1e-15 && tag.out_bw > 1e-15) {
                Time t_bl_drain = tag.backlog / tag.out_bw;
                if (t_bl_drain > 1e-15 && t_bl_drain < 1e6) {
                    int32_t idx = findBacklogDrainTimerIndex(port_id);
                    if (idx < 0 || t_bl_drain < backlogDrainTimers[idx]) {
                        setBacklogDrainTimer(port_id, t_bl_drain);
                    }
                }
            }
        }
    }
}

void Sim::pfcThresholdDetectSystem(Engine &ctx)
{
    if (enablePfc == 0) {
        return;
    }

    if (pfcEgress != 0) {
        return;
    }

    int32_t ingress_check[MAX_TOPO_PORTS] {};
    int32_t num_ingress_check = 0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none() || ctx.get<DirtyPort>(port_e).isDirty == 0) {
            continue;
        }
        for (int32_t i = 0; i < numTagIndexEntries; i++) {
            if (tagIndex[i].port_id != port_id) {
                continue;
            }
            FlowTagState &tag = ctx.get<FlowTagState>(tagIndex[i].entity);
            int32_t ip = tag.ingress_port_id;
            if (ip >= 0) {
                bool dup = false;
                for (int32_t j = 0; j < num_ingress_check; j++) {
                    if (ingress_check[j] == ip) {
                        dup = true;
                        break;
                    }
                }
                if (!dup && num_ingress_check < MAX_TOPO_PORTS) {
                    ingress_check[num_ingress_check++] = ip;
                }
            }
        }
    }

    for (int32_t i = 0; i < num_ingress_check; i++) {
        int32_t ingress_port = ingress_check[i];
        if (ingress_port < 0 || ingress_port >= numPorts) {
            continue;
        }
        Entity ingress_e = portEntities[ingress_port];
        if (ingress_e == Entity::none()) {
            continue;
        }

        PortPfcConfig &cfg = ctx.get<PortPfcConfig>(ingress_e);
        if (cfg.pfc_enabled == 0) {
            continue;
        }
        PortPfcState &state = ctx.get<PortPfcState>(ingress_e);

        double buf_by_pri[PFC_MAX_PRIORITY] {};
        double net_rate_by_pri[PFC_MAX_PRIORITY] {};
        for (int32_t j = 0; j < numIngressTags; j++) {
            if (ingressTags[j].ingress_port_id != ingress_port) {
                continue;
            }
            Entity te = ingressTags[j].entity;
            if (te == Entity::none()) {
                continue;
            }
            FlowTagState &tag = ctx.get<FlowTagState>(te);
            if (tag.is_source != 0) {
                continue;
            }
            materializeBacklog(tag, now);
            int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
            buf_by_pri[pri] += tag.backlog;
            net_rate_by_pri[pri] += (tag.in_bw - tag.out_bw);
        }

        int32_t upstream_port = -1;
        if (ingress_port >= 0 && ingress_port < numPorts) {
            upstream_port = peerPort[ingress_port];
        }

        bool state_changed = false;
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            double buf = buf_by_pri[pri];
            double net_rate = net_rate_by_pri[pri];
            if (state.pause_active[pri] == 0 && buf >= cfg.xoff[pri] - 0.5) {
                state.pause_active[pri] = 1;
                state_changed = true;
                state.paused_upstream_count[pri] = 0;
                if (upstream_port >= 0) {
                    state.paused_upstreams[pri][state.paused_upstream_count[pri]++] = upstream_port;
                    int32_t detect_slot = findNodeSlot(portToNode[ingress_port]);
                    int32_t upstream_slot = findNodeSlot(portToNode[upstream_port]);
                    Time pfc_delay = defaultLinkDelay;
                    if (detect_slot >= 0 && upstream_slot >= 0 && linkDelays[detect_slot][upstream_slot] >= 0.0) {
                        pfc_delay = linkDelays[detect_slot][upstream_slot];
                    }
                    DelayedEvent ev {};
                    ev.t = computePropagationTime(pfc_delay);
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = upstream_port,
                        .source_port_id = ingress_port,
                        .priority = pri,
                        .paused = 1,
                    };
                    pushDelayedEvent(ev);
                }
                state.pfc_cnt[pri] += 1;
            } else if (state.pause_active[pri] != 0 && buf <= cfg.xon[pri] + 0.5) {
                state.pause_active[pri] = 0;
                state_changed = true;
                for (int32_t k = 0; k < state.paused_upstream_count[pri]; k++) {
                    int32_t up = state.paused_upstreams[pri][k];
                    int32_t detect_slot = findNodeSlot(portToNode[ingress_port]);
                    int32_t upstream_slot = findNodeSlot(portToNode[up]);
                    Time pfc_delay = defaultLinkDelay;
                    if (detect_slot >= 0 && upstream_slot >= 0 && linkDelays[detect_slot][upstream_slot] >= 0.0) {
                        pfc_delay = linkDelays[detect_slot][upstream_slot];
                    }
                    DelayedEvent ev {};
                    ev.t = computePropagationTime(pfc_delay);
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = up,
                        .source_port_id = ingress_port,
                        .priority = pri,
                        .paused = 0,
                    };
                    pushDelayedEvent(ev);
                }
                state.paused_upstream_count[pri] = 0;
            }

            if (state.pause_active[pri] == 0 && net_rate > 1.0) {
                double gap = cfg.xoff[pri] - buf;
                if (gap > 1e-9) {
                    double t_xoff = gap / net_rate;
                    if (t_xoff > 1e-9 && t_xoff < 1e6) {
                        setPfcPauseTimer(ingress_port, t_xoff);
                    }
                }
            } else if (state.pause_active[pri] != 0) {
                double effective_net = net_rate;
                if (effective_net >= -1e-15) {
                    double out_total = 0.0;
                    for (int32_t j = 0; j < numIngressTags; j++) {
                        if (ingressTags[j].ingress_port_id != ingress_port) {
                            continue;
                        }
                        Entity te = ingressTags[j].entity;
                        if (te != Entity::none()) {
                            FlowTagState &t = ctx.get<FlowTagState>(te);
                            if (t.priority == pri) {
                                out_total += t.out_bw;
                            }
                        }
                    }
                    if (out_total > 1e-15) {
                        effective_net = -out_total;
                    }
                }
                if (effective_net < -1.0) {
                    double gap = buf - cfg.xon[pri];
                    if (gap > 1e-9) {
                        double t_xon = gap / (-effective_net);
                        if (t_xon > 1e-9 && t_xon < 1e6) {
                            setPfcResumeTimer(ingress_port, t_xon);
                        }
                    }
                }
            }
        }

        if (state_changed) {
            clearPfcPauseTimer(ingress_port);
            clearPfcResumeTimer(ingress_port);
        }
    }
}

void Sim::downstreamEmitSystem(Engine &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none() || ctx.get<DirtyPort>(port_e).isDirty == 0) {
            continue;
        }

        for (int32_t i = 0; i < numTagIndexEntries; i++) {
            if (tagIndex[i].port_id != port_id) {
                continue;
            }
            Entity tag_e = tagIndex[i].entity;
            if (tag_e == Entity::none()) {
                continue;
            }
            FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            if (tag.next_port_id < 0) {
                continue;
            }
            if (tag.downstream_created == 0) {
                if (tag.out_bw > 1e-15) {
                    DelayedEvent ev {};
                    ev.t = computePropagationTimeForPort(tag.port_id, tag.next_port_id);
                    ev.type = DelayedEvent::Type::Arrival;
                    ev.arrival = FlowArrivalEv {
                        .port_id = tag.next_port_id,
                        .flow_id = tag.flow_id,
                        .size = 0.0,
                        .in_bw = tag.out_bw,
                        .is_source = 0,
                        .priority = tag.priority,
                    };
                    pushDelayedEvent(ev);
                    tag.downstream_created = 1;
                }
                continue;
            }
            if (tag.out_bw == tag.prev_out_bw) {
                continue;
            }
            DelayedEvent ev {};
            ev.t = computePropagationTimeForPort(tag.port_id, tag.next_port_id);
            ev.type = DelayedEvent::Type::BwUpdate;
            ev.bwupd = BwUpdateEv {
                .port_id = tag.next_port_id,
                .flow_id = tag.flow_id,
                .in_bw = tag.out_bw,
            };
            pushDelayedEvent(ev);
        }
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
    enableBuffer = cfg.enable_buffer;
    enablePfc = cfg.enable_pfc;
    pfcEgress = cfg.pfc_egress;
    defaultLinkDelay = cfg.default_link_delay;
    propagationInterval = cfg.propagation_interval;
    pfcXoffThreshold = cfg.pfc_xoff_threshold;
    pfcXonThreshold = cfg.pfc_xon_threshold;
    dtMin = cfg.dt_min;
    qosMode = cfg.qos_mode;
    for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
        priorWeights[i] = cfg.prior_weights[i];
    }
    loadTopo(ctx);
    loadFlow(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
