#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/ecs.hpp>

#include "types.hpp"
#include "init.hpp"

namespace madsimple {

class Engine;

constexpr int32_t MAX_TOPO_NODES = 8;
constexpr int32_t MAX_TOPO_LINKS = 16;
constexpr int32_t MAX_TOPO_PORTS = 32;
constexpr int32_t MAX_NODE_NEIGHBORS = 8;
constexpr int32_t MAX_FLOWS = 16;
constexpr int32_t MAX_PATH_NODES = 16;
constexpr int32_t MAX_ECMP_NEXT_HOPS = 8;
constexpr int32_t MAX_FLOW_ROUTE_STEPS = 16;

struct TopoNeighbor {
    NodeId neighbor_id = -1;
    madrona::Entity port_entity = madrona::Entity::none();
    int32_t port_id = -1;
};

struct TopoNodeState {
    NodeId id = -1;
    NodeType type = NodeType::Host;
    Bw port_bw = 100.0;
    int32_t num_neighbors = 0;
    TopoNeighbor neighbors[MAX_NODE_NEIGHBORS] {};
};

struct TopoLinkState {
    NodeId src = -1;
    NodeId dst = -1;
    Time delay = 0.01;
    Bw bandwidth = 0.0;
};

struct FlowRouteStep {
    int32_t port_id = -1;
    int32_t next_port_id = -1;
};

struct FlowRouteState {
    FlowId flow_id = -1;
    int32_t num_steps = 0;
    FlowRouteStep steps[MAX_FLOW_ROUTE_STEPS] {};
};

struct Sim : public madrona::WorldBase {
    struct Config {
        uint32_t maxEpisodeLength;
        bool enableViewer;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    void resetNetworkState();
    void loadTopo(Engine &ctx);
    void loadFlow(Engine &ctx);
    void buildHardcodedTopo(NodeDef *nodes,
                            int32_t &num_nodes,
                            LinkDef *links,
                            int32_t &num_links) const;
    void buildHardcodedFlows(FlowDef *flows, int32_t &num_flows) const;
    void computeRoutes();
    int32_t createPort(Engine &ctx, NodeId node_id, int32_t port_idx, Bw port_bw);
    int32_t findNodeSlot(NodeId node_id) const;
    int32_t findNeighborSlot(int32_t node_slot, NodeId neighbor_id) const;
    int32_t getPath(NodeId src,
                    NodeId dst,
                    FlowId flow_id,
                    NodeId *out_path,
                    int32_t max_path) const;
    void injectFlowDef(Engine &ctx, const FlowDef &flow);
    void schedulePendingFlows(Engine &ctx);
    int32_t lookupFlowRouteNext(FlowId flow_id, int32_t port_id) const;

    EpisodeManager *episodeMgr;
    const GridState *grid;
    uint32_t maxEpisodeLength;

    Time now;
    int32_t nextPortID;
    int32_t numTopoNodes;
    TopoNodeState topoNodes[MAX_TOPO_NODES];
    int32_t numTopoLinks;
    TopoLinkState topoLinks[MAX_TOPO_LINKS];
    int32_t numPorts;
    madrona::Entity portEntities[MAX_TOPO_PORTS];
    NodeId portToNode[MAX_TOPO_PORTS];
    int32_t peerPort[MAX_TOPO_PORTS];
    Time linkDelays[MAX_TOPO_NODES][MAX_TOPO_NODES];
    NodeId routeTable[MAX_TOPO_NODES][MAX_TOPO_NODES];
    int32_t ecmpCount[MAX_TOPO_NODES][MAX_TOPO_NODES];
    NodeId ecmpNextHops[MAX_TOPO_NODES][MAX_TOPO_NODES][MAX_ECMP_NEXT_HOPS];
    int32_t numFlowDefs;
    FlowDef flowDefs[MAX_FLOWS];
    int32_t numPendingFlows;
    FlowDef pendingFlows[MAX_FLOWS];
    int32_t numFlowRoutes;
    FlowRouteState flowRoutes[MAX_FLOWS];
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
