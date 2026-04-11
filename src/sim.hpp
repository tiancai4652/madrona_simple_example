#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/ecs.hpp>

#include "types.hpp"
#include "init.hpp"

namespace madsimple {

class Engine;

constexpr int32_t MAX_TOPO_NODES = 67;
constexpr int32_t MAX_TOPO_LINKS = 132;
constexpr int32_t MAX_TOPO_PORTS = 132;
constexpr int32_t MAX_NODE_NEIGHBORS = 33;
constexpr int32_t MAX_FLOWS = 448;
constexpr int32_t MAX_PATH_NODES = 16;
constexpr int32_t MAX_ECMP_NEXT_HOPS = 8;
constexpr int32_t MAX_FLOW_ROUTE_STEPS = 16;
constexpr int32_t MAX_DELAYED_EVENTS = 16384;
constexpr int32_t MAX_EVENTS_PER_STEP = 16384;
constexpr int32_t MAX_TAG_INDEX = 4096;
constexpr int32_t MAX_SOURCE_TAGS = 512;
constexpr int32_t MAX_INGRESS_TAGS = 4096;
constexpr int32_t MAX_FLOW_COMPLETIONS = 512;
constexpr int32_t QOS_NONE = 0;
constexpr int32_t QOS_SP = 1;
constexpr int32_t QOS_WRR = 2;

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

struct FlowArrivalEv {
    int32_t port_id = -1;
    FlowId flow_id = -1;
    Bytes size = 0.0;
    Bw in_bw = 0.0;
    int32_t is_source = 0;
    int32_t priority = 0;
};

struct BwUpdateEv {
    int32_t port_id = -1;
    FlowId flow_id = -1;
    Bw in_bw = 0.0;
};

struct PfcControlEv {
    int32_t target_port_id = -1;
    int32_t source_port_id = -1;
    int32_t priority = 0;
    int32_t paused = 0;
};

struct DelayedEvent {
    enum class Type : int32_t {
        Arrival,
        BwUpdate,
        PfcControl,
    } type = Type::Arrival;

    Time t = 0.0;
    FlowArrivalEv arrival {};
    BwUpdateEv bwupd {};
    PfcControlEv pfcctrl {};
};

struct TagIndexEntry {
    int32_t port_id = -1;
    FlowId flow_id = -1;
    madrona::Entity entity = madrona::Entity::none();
};

struct SourceTagEntry {
    FlowId flow_id = -1;
    madrona::Entity entity = madrona::Entity::none();
};

struct IngressTagEntry {
    int32_t ingress_port_id = -1;
    FlowId flow_id = -1;
    madrona::Entity entity = madrona::Entity::none();
};

struct FlowCompletionEntry {
    FlowId flow_id = -1;
    FlowCompletionRecord record {};
};

struct Sim : public madrona::WorldBase {
    struct Config {
        uint32_t maxEpisodeLength = 200;
        bool enableViewer = false;
        Time default_link_delay = 0.001;
        Time propagation_interval = 0.0;
        int32_t enable_buffer = 1;
        int32_t enable_pfc = 0;
        int32_t pfc_egress = 0;
        double pfc_xoff_threshold = 1e9;
        double pfc_xon_threshold = 0.5e9;
        double dt_min = 0.0;
        int32_t qos_mode = QOS_NONE;
        double prior_weights[PFC_MAX_PRIORITY] {};
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
    void injectFlowDef(const FlowDef &flow);
    void injectFlow(int32_t src_port_id, const FlowDef &flow);
    void schedulePendingFlows();
    void deliverEvents();
    void flowArrivalSystem(Engine &ctx);
    void bwUpdateIngressSystem(Engine &ctx);
    void pfcPropagateSystem(Engine &ctx);
    void portBandwidthAllocSystem(Engine &ctx, Time dt);
    void pfcThresholdDetectSystem(Engine &ctx);
    void downstreamEmitSystem(Engine &ctx);
    void clearDirtyPorts(Engine &ctx);
    Time chooseDT() const;
    void bufferUpdateSystem(Engine &ctx, Time dt);
    void flowProgressAndCleanupSystem(Engine &ctx, Time dt);
    int32_t lookupFlowRouteNext(FlowId flow_id, int32_t port_id) const;
    madrona::Entity findTag(int32_t port_id, FlowId flow_id) const;
    madrona::Entity createTagOnPort(Engine &ctx,
                                    int32_t port_id,
                                    FlowId flow_id,
                                    Bw in_bw,
                                    Bytes size,
                                    bool is_source,
                                    int32_t priority);
    void destroyTag(Engine &ctx,
                    madrona::Entity tag_entity,
                    bool propagate_cleanup,
                    Time logical_now);
    void recordFlowCompletion(FlowId flow_id, Time end_time);
    void materializeBacklog(FlowTagState &tag, Time at_time);
    void materializeRemaining(FlowTagState &tag, Time at_time);
    void materializeBufCnt(PortBuffer &port_buf, Time at_time);
    void alignChunksWithBufCnt(PriorityBuffer &pb);
    double drainBufferChunks(PriorityBuffer &pb, double drain_bytes);
    Time computePropagationTimeAt(Time base_time, Time link_delay) const;
    Time computePropagationTime(Time link_delay) const;
    Time computePropagationTimeForPort(int32_t src_port_id, int32_t dst_port_id) const;
    void pushDelayedEvent(const DelayedEvent &ev);
    int32_t findSourceTagIndex(FlowId flow_id) const;
    int32_t findIngressTagIndex(int32_t ingress_port_id, FlowId flow_id) const;
    int32_t findBacklogDrainTimerIndex(int32_t port_id) const;
    int32_t findPfcPauseTimerIndex(int32_t ingress_port_id) const;
    int32_t findPfcResumeTimerIndex(int32_t ingress_port_id) const;
    void setBacklogDrainTimer(int32_t port_id, Time t);
    void setPfcPauseTimer(int32_t ingress_port_id, Time t);
    void setPfcResumeTimer(int32_t ingress_port_id, Time t);
    void clearBacklogDrainTimer(int32_t port_id);
    void clearPfcPauseTimer(int32_t ingress_port_id);
    void clearPfcResumeTimer(int32_t ingress_port_id);

    EpisodeManager *episodeMgr;
    const GridState *grid;
    const NetworkInit *network;
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

    int32_t numDelayedEvents;
    DelayedEvent delayedEvents[MAX_DELAYED_EVENTS];
    int32_t numInboxArrival;
    FlowArrivalEv inboxArrival[MAX_EVENTS_PER_STEP];
    int32_t numInboxBwUpdate;
    BwUpdateEv inboxBwUpdate[MAX_EVENTS_PER_STEP];
    int32_t numInboxPfc;
    PfcControlEv inboxPfc[MAX_EVENTS_PER_STEP];
    int32_t numTagIndexEntries;
    TagIndexEntry tagIndex[MAX_TAG_INDEX];
    int32_t numSourceTags;
    SourceTagEntry sourceTags[MAX_SOURCE_TAGS];
    int32_t numIngressTags;
    IngressTagEntry ingressTags[MAX_INGRESS_TAGS];
    int32_t numFlowCompletions;
    FlowCompletionEntry flowCompletions[MAX_FLOW_COMPLETIONS];
    int32_t enableBuffer;
    int32_t enablePfc;
    int32_t pfcEgress;
    Time defaultLinkDelay;
    Time propagationInterval;
    double pfcXoffThreshold;
    double pfcXonThreshold;
    double dtMin;
    int32_t qosMode;
    double priorWeights[PFC_MAX_PRIORITY];
    Time cachedNextDrainTime;
    int32_t cachedDrainPortID;
    Time cachedNextFinishTime;
    int32_t numBacklogDrainTimers;
    int32_t backlogDrainPortIDs[MAX_TOPO_PORTS];
    Time backlogDrainTimers[MAX_TOPO_PORTS];
    int32_t numPfcPauseTimers;
    int32_t pfcPausePortIDs[MAX_TOPO_PORTS];
    Time pfcPauseTimers[MAX_TOPO_PORTS];
    int32_t numPfcResumeTimers;
    int32_t pfcResumePortIDs[MAX_TOPO_PORTS];
    Time pfcResumeTimers[MAX_TOPO_PORTS];
    int32_t numLastDirtyPortIDs;
    int32_t lastDirtyPortIDs[MAX_TOPO_PORTS];
    Time nextDT;
    uint64_t systemLogStep;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
