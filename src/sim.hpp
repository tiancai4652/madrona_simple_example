#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/ecs.hpp>

#include "types.hpp"
#include "init.hpp"

namespace madsimple {

class Engine;

// constexpr int32_t MAX_TOPO_NODES = 67;
// constexpr int32_t MAX_TOPO_LINKS = 132;
// constexpr int32_t MAX_TOPO_PORTS = 132;
// constexpr int32_t MAX_NODE_NEIGHBORS = 33;
// constexpr int32_t MAX_FLOWS = 448;
// constexpr int32_t MAX_PATH_NODES = 16;
// constexpr int32_t MAX_ECMP_NEXT_HOPS = 8;
// constexpr int32_t MAX_FLOW_ROUTE_STEPS = 16;
// constexpr int32_t MAX_DELAYED_EVENTS = 16384;
// constexpr int32_t MAX_EVENTS_PER_STEP = 16384;
// constexpr int32_t MAX_TAG_INDEX = 4096;
// constexpr int32_t MAX_SOURCE_TAGS = 512;
// constexpr int32_t MAX_INGRESS_TAGS = 4096;
// MAX_FLOW_COMPLETIONS moved to types.hpp so the FlowCompletionBuf export
// component can size itself without including sim.hpp.
constexpr int32_t QOS_NONE = 0;
constexpr int32_t QOS_SP = 1;
constexpr int32_t QOS_WRR = 2;


constexpr int32_t MAX_TOPO_NODES = 1152;
constexpr int32_t MAX_TOPO_LINKS = 3200;
constexpr int32_t MAX_TOPO_PORTS = 3200;
constexpr int32_t MAX_NODE_NEIGHBORS = 56;
constexpr int32_t MAX_FLOWS = 68608;
constexpr int32_t MAX_PATH_NODES = 6;
constexpr int32_t MAX_ECMP_NEXT_HOPS = 36;
constexpr int32_t MAX_FLOW_ROUTE_STEPS = 6;
constexpr int32_t MAX_DELAYED_EVENTS = 131072;
constexpr int32_t MAX_EVENTS_PER_STEP = 1024;
constexpr int32_t MAX_TAG_INDEX = 258048;
constexpr int32_t MAX_SOURCE_TAGS = 64512;
constexpr int32_t MAX_INGRESS_TAGS = 258048;
// constexpr int32_t MAX_FLOW_COMPLETIONS = 68608;

struct TopoNeighbor {
    NodeId neighbor_id = -1;
    int32_t neighbor_slot = -1;
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

// FlowArrivalEv / BwUpdateEv / PfcControlEv / DelayedEvent were moved to
// types.hpp (phase C) so PortOutbox can embed them in the Port archetype.

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
        int32_t perf_fct_only = 1;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static MADRONA_NO_INLINE void setupTasks(
        madrona::TaskGraphManager &taskgraph_mgr,
        const Config &cfg);

    MADRONA_NO_INLINE Sim(Engine &ctx, const Config &cfg,
                          const WorldInit &init);

    MADRONA_NO_INLINE void resetNetworkState();
    MADRONA_NO_INLINE void loadTopo(Engine &ctx);
    MADRONA_NO_INLINE void loadFlow(Engine &ctx);
    void buildHardcodedTopo(NodeDef *nodes,
                            int32_t &num_nodes,
                            LinkDef *links,
                            int32_t &num_links) const;
    void buildHardcodedFlows(FlowDef *flows, int32_t &num_flows) const;
    void computeRoutes();
    MADRONA_NO_INLINE int32_t createPort(Engine &ctx,
                                         NodeId node_id,
                                         int32_t port_idx,
                                         Bw port_bw);
    int32_t findNodeSlot(NodeId node_id) const;
    int32_t findNeighborSlot(int32_t node_slot, NodeId neighbor_id) const;
    int32_t getPath(NodeId src,
                    NodeId dst,
                    FlowId flow_id,
                    NodeId *out_path,
                    int32_t max_path) const;
    MADRONA_NO_INLINE void injectFlowDef(const FlowDef &flow);
    MADRONA_NO_INLINE void injectFlow(int32_t src_port_id,
                                      const FlowDef &flow);
    MADRONA_NO_INLINE void schedulePendingFlows();
    MADRONA_NO_INLINE void deliverEvents(madrona::Context &ctx);
    MADRONA_NO_INLINE Time chooseDT() const;
    MADRONA_NO_INLINE void flowProgressAndCleanupSystem(
        madrona::Context &ctx, Time dt);
    MADRONA_NO_INLINE void progressFinishedSources(
        madrona::Context &ctx,
        Time dt,
        Time next_now,
        int32_t &finished_source_count,
        int32_t &emitted_cleanup_count);
    MADRONA_NO_INLINE void progressBacklogDrainTimers(
        madrona::Context &ctx,
        Time dt);
    MADRONA_NO_INLINE void markIngressTagsDirty(
        madrona::Context &ctx,
        int32_t ingress_port);
    MADRONA_NO_INLINE void progressPfcTimers(
        madrona::Context &ctx,
        Time dt);
    MADRONA_NO_INLINE void markBufferedPortsDirty(
        madrona::Context &ctx);
    MADRONA_NO_INLINE void progressExhaustedPfcState(
        madrona::Context &ctx,
        Time dt);

    inline bool perfFCTOnlyEnabled() const
    {
        return perfFCTOnly != 0;
    }

    inline bool traceModeEnabled() const
    {
        return perfFCTOnly == 0;
    }

    // Phase B.1 singleton: consumes PortTraceLast.was_dirty_at_clear written
    // by the per-Port clearDirtyOnePortSystem to rebuild lastDirtyPortIDs in
    // port_id ascending order.
    MADRONA_NO_INLINE void snapshotDirtyPorts(madrona::Context &ctx);

    // Phase B.2: per-Port bandwidth allocation worker. Reads global
    // topology/tagIndex read-only and writes only to the port's own
    // components or the supplied hint / cleanup / trace buffers. All
    // cross-port state (cachedNextDrainTime / cachedDrainPortID /
    // cachedNextFinishTime / backlogDrainTimers / destroyTag effects) is
    // deferred to the dedicated singleton flush systems below.
    MADRONA_NO_INLINE void allocOnePort(madrona::Context &ctx,
                                        int32_t port_id,
                                        PortState &port_state,
                                        PortBuffer &port_buf,
                                        DirtyPort &dirty,
                                        PortPfcConfig &pfc_cfg,
                                        PortPfcState &pfc_state,
                                        PortCachedHints &hints,
                                        PortDrainHint &drain_hint,
                                        PortCleanup &cleanup,
                                        PortTraceLast &trace,
                                        PortTagList &tag_list);

    // Phase B.2 singletons (driven by SimDriverArch). They walk
    // portEntities[] in port_id ascending order so per-frame outputs are
    // deterministic even when allocOnePort runs in parallel on GPU.
    MADRONA_NO_INLINE void reducePortCachedHints(madrona::Context &ctx);
    MADRONA_NO_INLINE void flushPortDrainHints(madrona::Context &ctx);
    MADRONA_NO_INLINE void flushPortTagCleanup(madrona::Context &ctx);
    MADRONA_NO_INLINE void logAllocTraces(madrona::Context &ctx);

    // Phase B.3: per-Port buffer-advance worker. Computes the legacy
    // processPorts decision locally (lastDirtyPortIDs / cachedDrainPortID /
    // backlogDrainTimers / any-priority-empty heuristic) and, if the port
    // is eligible, advances that port's PortBuffer / FlowTagState in
    // place. Tags queued for destruction land in PortCleanup and are
    // flushed sequentially by flushBufferTagCleanup below.
    MADRONA_NO_INLINE void advanceOnePortBuffer(madrona::Context &ctx,
                                                int32_t port_id,
                                                Time dt,
                                                PortState &port_state,
                                                PortBuffer &port_buf,
                                                DirtyPort &dirty,
                                                PortPfcState &pfc_state,
                                                PortCleanup &cleanup,
                                                PortTraceLast &trace,
                                                PortTagList &tag_list);

    // Phase B.3 singletons. flushBufferTagCleanup replays the buffer-phase
    // destroyTag calls in port_id ascending order; logBufferTraces sums
    // the per-port buffer summary fields for the "buffer" scope log.
    MADRONA_NO_INLINE void flushBufferTagCleanup(madrona::Context &ctx);
    MADRONA_NO_INLINE void logBufferTraces(madrona::Context &ctx);

    // Phase C: per-Port pfcDetect worker. Computes the egress- or
    // ingress-mode PFC detect body for one port. Cross-port side-effects
    // (pushDelayedEvent, setPfcPauseTimer/setPfcResumeTimer/clear*) are
    // captured in PortOutbox and PortPfcState.want_* and flushed by the
    // SimDriver singletons that follow.
    MADRONA_NO_INLINE void pfcDetectOnePort(madrona::Context &ctx,
                                            int32_t port_id,
                                            PortState &port_state,
                                            PortBuffer &port_buf,
                                            DirtyPort &dirty,
                                            PortPfcConfig &pfc_cfg,
                                            PortPfcState &pfc_state,
                                            PortOutbox &outbox,
                                            PortTraceLast &trace,
                                            PortTagList &tag_list);

    // Helper split of pfcDetectOnePort for the egress and ingress branches.
    // Kept as two separate non-inlined functions so NVRTC + ptxas optimize
    // each branch's control-flow graph independently; empirically this
    // prevents the combinatorial blow-up that hangs `-dlto -dopt=on
    // --extra-device-vectorization` when both branches live in the same
    // function body. Behaviour is 1:1 with the original monolithic
    // pfcDetectOnePort; see sim_systems_pfc.cpp for the full contract.
    MADRONA_NO_INLINE void pfcDetectOnePortEgress(
        madrona::Context &ctx,
        int32_t port_id,
        DirtyPort &dirty,
        PortPfcConfig &cfg,
        PortPfcState &state,
        PortOutbox &outbox,
        PortTraceLast &trace,
        PortTagList &tag_list);

    MADRONA_NO_INLINE void pfcDetectOnePortIngress(
        madrona::Context &ctx,
        int32_t port_id,
        PortPfcConfig &cfg,
        PortPfcState &state,
        PortOutbox &outbox,
        PortTraceLast &trace);

    // Phase C: per-Port emit worker. Mirrors legacy downstreamEmitSystem
    // but for a single port. Pushes Arrival/BwUpdate DelayedEvents into
    // the port's PortOutbox only, never into Sim::delayedEvents directly.
    MADRONA_NO_INLINE void emitOnePort(madrona::Context &ctx,
                                       int32_t port_id,
                                       PortState &port_state,
                                       DirtyPort &dirty,
                                       PortOutbox &outbox,
                                       PortTraceLast &trace,
                                       PortTagList &tag_list);

    // Phase C singletons. They walk portEntities[] in port_id ascending
    // order and fold per-Port PortOutbox / PortPfcState.want_* / trace
    // fields back into the global Sim arrays deterministically.
    MADRONA_NO_INLINE void flushPortOutbox(madrona::Context &ctx);
    MADRONA_NO_INLINE void flushPortPfcTimers(madrona::Context &ctx);
    MADRONA_NO_INLINE void logPfcDetectTraces(madrona::Context &ctx);
    MADRONA_NO_INLINE void logEmitTraces(madrona::Context &ctx);

    // Phase E: per-Port ingress-chain workers. They consume the target
    // port's PortInbox (dispatched by deliverEvents) and only write to
    // the port's own components plus the deferred PortCreateList /
    // PortCleanup / PortCompletionList / PortOutbox; all global mutation
    // is flushed by the singletons below.
    MADRONA_NO_INLINE void pfcPropagateOnePort(madrona::Context &ctx,
                                               int32_t port_id,
                                               PortState &port_state,
                                               PortPfcState &pfc_state,
                                               DirtyPort &dirty,
                                               PortInbox &inbox,
                                               PortTraceLast &trace);
    MADRONA_NO_INLINE void flowArrivalOnePort(madrona::Context &ctx,
                                              int32_t port_id,
                                              PortState &port_state,
                                              DirtyPort &dirty,
                                              PortInbox &inbox,
                                              PortTagList &tag_list,
                                              PortCreateList &create_list,
                                              PortTraceLast &trace);
    MADRONA_NO_INLINE void bwUpdateOnePort(madrona::Context &ctx,
                                           int32_t port_id,
                                           PortState &port_state,
                                           PortBuffer &port_buf,
                                           DirtyPort &dirty,
                                           PortInbox &inbox,
                                           PortTagList &tag_list,
                                           PortCreateList &create_list,
                                           PortCleanup &cleanup,
                                           PortOutbox &outbox,
                                           PortCompletionList &completions,
                                           PortTraceLast &trace);

    // Phase E singletons. flushPortInboxReset zeroes every port's
    // PortInbox before deliverEvents writes into it; dispatchEvents (the
    // new deliverEvents variant that owns ctx) walks delayedEvents in
    // arrival order and writes each due event into the target port's
    // PortInbox. flushTagCreate / flushFlowCompletion materialise the
    // deferred per-Port requests in port_id ascending order.
    void dispatchEvents(madrona::Context &ctx);
    MADRONA_NO_INLINE void flushTagCreate(madrona::Context &ctx);
    MADRONA_NO_INLINE void flushFlowCompletion(madrona::Context &ctx);
    MADRONA_NO_INLINE void logIngressChain(madrona::Context &ctx);

    MADRONA_NO_INLINE int32_t lookupFlowRouteNext(
        FlowId flow_id, int32_t port_id) const;
    MADRONA_NO_INLINE madrona::Entity findTag(
        int32_t port_id, FlowId flow_id) const;
    MADRONA_NO_INLINE madrona::Entity createTagOnPort(
        madrona::Context &ctx,
        int32_t port_id,
        FlowId flow_id,
        Bw in_bw,
        Bytes size,
        bool is_source,
        int32_t priority);
    MADRONA_NO_INLINE void destroyTag(madrona::Context &ctx,
                                      madrona::Entity tag_entity,
                                      bool propagate_cleanup,
                                      Time logical_now);
    MADRONA_NO_INLINE void recordFlowCompletion(
        FlowId flow_id, Time end_time);
    MADRONA_NO_INLINE void removeFlowDef(FlowId flow_id);
    MADRONA_NO_INLINE void removeFlowRoute(FlowId flow_id);
    MADRONA_NO_INLINE void materializeBacklog(FlowTagState &tag, Time at_time);
    MADRONA_NO_INLINE void materializeRemaining(FlowTagState &tag, Time at_time);
    MADRONA_NO_INLINE void materializeBufCnt(PortBuffer &port_buf,
                                             Time at_time);
    MADRONA_NO_INLINE void alignChunksWithBufCnt(PriorityBuffer &pb);
    MADRONA_NO_INLINE double drainBufferChunks(PriorityBuffer &pb,
                                               double drain_bytes);
    MADRONA_NO_INLINE Time computePropagationTimeAt(
        Time base_time, Time link_delay) const;
    MADRONA_NO_INLINE Time computePropagationTime(Time link_delay) const;
    MADRONA_NO_INLINE Time computePropagationTimeForPort(
        int32_t src_port_id, int32_t dst_port_id) const;
    MADRONA_NO_INLINE void pushDelayedEvent(const DelayedEvent &ev);
    MADRONA_NO_INLINE int32_t findSourceTagIndex(FlowId flow_id) const;
    MADRONA_NO_INLINE int32_t findIngressTagIndex(
        int32_t ingress_port_id, FlowId flow_id) const;
    MADRONA_NO_INLINE int32_t findBacklogDrainTimerIndex(
        int32_t port_id) const;
    MADRONA_NO_INLINE int32_t findPfcPauseTimerIndex(
        int32_t ingress_port_id) const;
    MADRONA_NO_INLINE int32_t findPfcResumeTimerIndex(
        int32_t ingress_port_id) const;
    MADRONA_NO_INLINE void setBacklogDrainTimer(int32_t port_id, Time t);
    MADRONA_NO_INLINE void setPfcPauseTimer(int32_t ingress_port_id, Time t);
    MADRONA_NO_INLINE void setPfcResumeTimer(int32_t ingress_port_id, Time t);
    MADRONA_NO_INLINE void clearBacklogDrainTimer(int32_t port_id);
    MADRONA_NO_INLINE void clearPfcPauseTimer(int32_t ingress_port_id);
    MADRONA_NO_INLINE void clearPfcResumeTimer(int32_t ingress_port_id);

    EpisodeManager *episodeMgr;
    const GridState *grid;
    const NetworkInit *network;
    uint32_t maxEpisodeLength;

    Time now;
    int32_t nextPortID;
    int32_t numTopoNodes;
    NodeId nodeLookupBase = 0;
    int32_t nodeLookupSpan = 0;
    int32_t nodeSlotLookup[MAX_TOPO_NODES];
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
    // computeRoutes() reuses a single BFS frontier / distance buffer per
    // destination to keep GPU world init O(numTopoNodes * (V + E)) instead of
    // materializing and clearing a full MAX_TOPO_NODES x MAX_TOPO_NODES
    // scratch matrix inside the device-side constructor.
    int32_t bfsDist[MAX_TOPO_NODES];
    int32_t bfsQueue[MAX_TOPO_NODES];
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
    // GPU megakernel stack must stay small; progressFinishedSources reuses this
    // world-owned scratch buffer instead of materializing MAX_SOURCE_TAGS on
    // every thread stack.
    madrona::Entity finishedSourceScratch[MAX_SOURCE_TAGS];
    int32_t numIngressTags;
    IngressTagEntry ingressTags[MAX_INGRESS_TAGS];
    int32_t numFlowCompletions;
    FlowCompletionEntry flowCompletions[MAX_FLOW_COMPLETIONS];
    // Large per-port scratch queues live on Sim instead of the Port
    // archetype to keep GPU initWorlds / createPort archetype creation
    // lighter while preserving deterministic port_id indexing.
    DirtyPort portDirtyStates[MAX_TOPO_PORTS];
    PortCleanup portCleanups[MAX_TOPO_PORTS];
    PortOutbox portOutboxes[MAX_TOPO_PORTS];
    PortTagList portTagLists[MAX_TOPO_PORTS];
    PortInbox portInboxes[MAX_TOPO_PORTS];
    PortCreateList portCreateLists[MAX_TOPO_PORTS];
    PortCompletionList portCompletionLists[MAX_TOPO_PORTS];
    PortPfcConfig portPfcConfigs[MAX_TOPO_PORTS];
    PortPfcState portPfcStates[MAX_TOPO_PORTS];
    PortCachedHints portCachedHints[MAX_TOPO_PORTS];
    PortDrainHint portDrainHints[MAX_TOPO_PORTS];
    PortTraceLast portTraceLasts[MAX_TOPO_PORTS];
    int32_t enableBuffer;
    int32_t enablePfc;
    int32_t pfcEgress;
    int32_t perfFCTOnly;
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
