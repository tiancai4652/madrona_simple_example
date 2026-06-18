#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/ecs.hpp>

#include <limits>

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
constexpr int32_t MAX_TOPO_NODES = 7488;
constexpr int32_t MAX_TOPO_LINKS = 21632;
constexpr int32_t MAX_TOPO_PORTS = 21632;
constexpr int32_t MAX_NODE_NEIGHBORS = 232;
constexpr int32_t MAX_FLOWS = 956416;
constexpr int32_t MAX_PATH_NODES = 6;
constexpr int32_t MAX_ECMP_NEXT_HOPS = 20;
constexpr int32_t MAX_FLOW_ROUTE_STEPS = 6;
constexpr int32_t MAX_DELAYED_EVENTS = 4329472;
constexpr int32_t MAX_EVENTS_PER_STEP = 689152;
constexpr int32_t MAX_TAG_INDEX = 3641344;
constexpr int32_t MAX_SOURCE_TAGS = 910336;
// constexpr int32_t MAX_FLOW_COMPLETIONS = 68608;

struct TopoNeighbor {
    NodeId neighbor_id = -1;
    int32_t neighbor_slot = -1;
    madrona::Entity port_entity = madrona::Entity::none();
    int32_t port_id = -1;
    Time delay = -1.0;
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

struct FlowRuntimeState {
    int32_t pending = 1;
    int32_t active = 0;
    int32_t completed = 0;
    int32_t route_active = 0;
    FlowCompletionRecord completion_record {};
};

// Per-flow scratch for the schedule split. The FlowMeta worker fills this
// with a precomputed port path and first delayed event for flows that are
// both inside the current pending window and ready at sim.now; the singleton
// schedule flush then commits the scratch in input order so delayed-event
// insertion order and counters stay deterministic.
struct FlowScheduleState {
    int32_t flow_order = -1;
    int32_t ready_now = 0;
    int32_t prepared = 0;
    int32_t port_path_len = 0;
    int32_t port_path[MAX_FLOW_ROUTE_STEPS + 1] {};
    DelayedEvent prepared_event {};
};

struct FlowMeta : public madrona::Archetype<
    FlowDef,
    FlowRouteState,
    FlowRuntimeState,
    FlowScheduleState
> {};

// FlowArrivalEv / BwUpdateEv / PfcControlEv / DelayedEvent were moved to
// types.hpp (phase C) so PortOutbox can embed them in the Port archetype.

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
    Time getLinkDelay(NodeId src, NodeId dst) const;
    Time getPortLinkDelay(int32_t src_port_id, int32_t dst_port_id) const;
    int32_t flowLookupIndex(FlowId flow_id) const;
    int32_t findTagLookupSlot(const PortTagLookup &lookup,
                              FlowId flow_id) const;
    void insertTagLookup(PortTagLookup &lookup,
                         FlowId flow_id,
                         madrona::Entity entity);
    void removeTagLookup(PortTagLookup &lookup, FlowId flow_id);
    madrona::Entity findFlowMetaEntity(
        madrona::Context &ctx, FlowId flow_id) const;
    const FlowDef *getFlowDef(madrona::Context &ctx, FlowId flow_id) const;
    int32_t getPath(NodeId src,
                    NodeId dst,
                    FlowId flow_id,
                    NodeId *out_path,
                    int32_t max_path) const;
    MADRONA_NO_INLINE bool injectFlowDef(madrona::Context &ctx,
                                         madrona::Entity flow_entity,
                                         DelayedEvent &out_ev);
    MADRONA_NO_INLINE bool prepareFlowScheduleState(
        const FlowDef &flow,
        FlowScheduleState &schedule_state) const;
    MADRONA_NO_INLINE bool buildFlowArrivalEvent(int32_t src_port_id,
                                                 const FlowDef &flow,
                                                 DelayedEvent &out_ev) const;
    MADRONA_NO_INLINE void preparePendingFlowMeta(
        madrona::Context &ctx,
        const FlowDef &flow,
        const FlowRuntimeState &runtime,
        FlowScheduleState &schedule_state) const;
    MADRONA_NO_INLINE void schedulePendingFlows(madrona::Context &ctx);
    MADRONA_NO_INLINE void deliverEvents(madrona::Context &ctx);
    MADRONA_NO_INLINE void deliverEventsOnePort(
        madrona::Context &ctx,
        int32_t port_id,
        PortDelayedQueue &queue,
        PortInbox &inbox,
        PortTraceLast &trace);
    MADRONA_NO_INLINE void finishDeliverEvents(madrona::Context &ctx);
    MADRONA_NO_INLINE Time chooseDT(madrona::Context &ctx) const;
    MADRONA_NO_INLINE void flowProgressAndCleanupSystem(
        madrona::Context &ctx, Time dt);
    MADRONA_NO_INLINE void progressFinishedSourcesOnePort(
        madrona::Context &ctx,
        Time dt,
        Time next_now,
        int32_t port_id,
        PortState &port_state,
        PortTagList &tag_list,
        const PortSourceTagList &source_tag_list,
        PortCachedHints &hints,
        PortFinishedSourceList &finished_list,
        PortTraceLast &trace,
        PortOutbox &outbox);
    MADRONA_NO_INLINE void cleanupFinishedSourcesOnePort(
        madrona::Context &ctx,
        Time next_now,
        int32_t port_id,
        PortCachedHints &hints,
        PortFinishedSourceList &finished_list);
    MADRONA_NO_INLINE void progressBacklogDrainTimers(
        madrona::Context &ctx,
        Time dt);
    MADRONA_NO_INLINE void progressBacklogDrainTimerOnePort(
        PortTimers &timers,
        DirtyPort &dirty,
        Time dt);
    MADRONA_NO_INLINE void progressPfcTimerOnePort(
        madrona::Context &ctx,
        int32_t ingress_port_id,
        PortTimers &timers,
        const IngressTagList &ingress_list,
        PortDirtyMarkList &dirty_marks,
        Time dt);
    MADRONA_NO_INLINE void flushDirtyPortMarks(madrona::Context &ctx);
    MADRONA_NO_INLINE void prepareProgressState(madrona::Context &ctx);
    MADRONA_NO_INLINE void markBufferedPortDirtyOnePort(
        PortBuffer &port_buf,
        DirtyPort &dirty,
        PortTraceLast &trace) const;
    MADRONA_NO_INLINE void finishProgressState(
        madrona::Context &ctx,
        Time dt);
    MADRONA_NO_INLINE void snapshotDirtyOnePort(
        int32_t port_id,
        const PortDelayedQueue &queue,
        const PortTimers &timers,
        PortTraceLast &trace) const;
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

    static constexpr Time timerInactiveSentinel()
    {
        return std::numeric_limits<Time>::max();
    }

    inline bool timerIsActive(Time t) const
    {
        return t < timerInactiveSentinel();
    }

    // Phase B.1 singleton: emits the clear-summary log and computes nextDT
    // after clearDirtyOnePortSystem has snapshotted each port's
    // was_dirty_at_clear flag locally.
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
    // processPorts decision locally (this port's was_dirty_at_clear /
    // cachedDrainPortID / backlogDrainTimers / any-priority-empty
    // heuristic) and, if the port is eligible, advances that port's
    // PortBuffer / FlowTagState in place. Tags queued for destruction land
    // in PortCleanup and are flushed sequentially by
    // flushBufferTagCleanup below.
    MADRONA_NO_INLINE void advanceOnePortBuffer(madrona::Context &ctx,
                                                int32_t port_id,
                                                Time dt,
                                                PortState &port_state,
                                                PortBuffer &port_buf,
                                                DirtyPort &dirty,
                                                PortPfcState &pfc_state,
                                                PortTimers &timers,
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
                                            PortTagList &tag_list,
                                            IngressTagList &ingress_list);

    // Helper split of pfcDetectOnePort for the egress and ingress branches.
    // Kept as two separate non-inlined functions so NVRTC + ptxas optimize
    // each branch's control-flow graph independently; empirically this
    // prevents the combinatorial blow-up that hangs `-dlto -dopt=on
    // --extra-device-vectorization` when both branches live in the same
    // function body. Behaviour is 1:1 with the original monolithic
    // pfcDetectOnePort; see sim_systems_pfc.cpp for the full contract.
    MADRONA_NO_INLINE void markPfcIngressCheckTargets(
        madrona::Context &ctx);

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
        PortTraceLast &trace,
        IngressTagList &ingress_list);

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
    MADRONA_NO_INLINE void logPfcDebugTraces(madrona::Context &ctx);
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
                                              PortSourceTagList &source_tag_list,
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
    // PortInbox. Per-Port tag creation is now split into a local
    // materialize worker plus a tiny singleton that replays only the
    // deferred ingress-list links in port_id ascending order.
    void dispatchEvents(madrona::Context &ctx);
    MADRONA_NO_INLINE void materializeTagCreateOnePort(
        madrona::Context &ctx,
        int32_t port_id,
        DirtyPort &dirty,
        PortCreateList &create_list,
        PortIngressLinkList &ingress_links,
        PortTraceLast &trace);
    MADRONA_NO_INLINE void materializeTagCleanupOnePort(
        madrona::Context &ctx,
        int32_t port_id,
        PortCleanup &cleanup,
        PortIngressUnlinkList &ingress_unlinks,
        PortCompletionList &completions,
        PortOutbox &outbox,
        Time logical_now);
    MADRONA_NO_INLINE void flushIngressTagLinks(madrona::Context &ctx);
    MADRONA_NO_INLINE void flushIngressTagUnlinks(madrona::Context &ctx);
    MADRONA_NO_INLINE void flushFlowCompletion(madrona::Context &ctx);
    MADRONA_NO_INLINE void logIngressChain(madrona::Context &ctx);
    MADRONA_NO_INLINE void refreshTagCounters(madrona::Context &ctx);

    MADRONA_NO_INLINE int32_t lookupFlowRouteNext(
        madrona::Context &ctx,
        FlowId flow_id,
        int32_t port_id) const;
    MADRONA_NO_INLINE int32_t lookupFlowIngressPort(
        madrona::Context &ctx,
        FlowId flow_id,
        int32_t port_id) const;
    MADRONA_NO_INLINE madrona::Entity findTag(
        madrona::Context &ctx, int32_t port_id, FlowId flow_id) const;
    MADRONA_NO_INLINE bool upstreamTagAlive(
        madrona::Context &ctx,
        const FlowTagState &tag) const;
    MADRONA_NO_INLINE madrona::Entity createTagOnPort(
        madrona::Context &ctx,
        int32_t port_id,
        FlowId flow_id,
        Bw in_bw,
        Bytes size,
        bool is_source,
        int32_t priority,
        bool link_ingress = true);
    MADRONA_NO_INLINE void destroyTag(madrona::Context &ctx,
                                      madrona::Entity tag_entity,
                                      bool propagate_cleanup,
                                      Time logical_now);
    MADRONA_NO_INLINE bool destroyTagCollectCleanupEvent(
        madrona::Context &ctx,
        madrona::Entity tag_entity,
        bool propagate_cleanup,
        Time logical_now,
        DelayedEvent &out_ev);
    MADRONA_NO_INLINE bool destroyTagMaterializeOnePort(
        madrona::Context &ctx,
        madrona::Entity tag_entity,
        bool propagate_cleanup,
        Time logical_now,
        PortIngressUnlinkList &ingress_unlinks,
        PortCompletionList &completions,
        PortOutbox &outbox);
    MADRONA_NO_INLINE void recordFlowCompletion(
        madrona::Context &ctx, FlowId flow_id, Time end_time);
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
    MADRONA_NO_INLINE void pushDelayedEvent(
        madrona::Context &ctx,
        const DelayedEvent &ev);
    MADRONA_NO_INLINE void pushDelayedEventsBatch(
        madrona::Context &ctx,
        const DelayedEvent *events,
        int32_t count);
    MADRONA_NO_INLINE void setBacklogDrainTimer(PortTimers &timers, Time t);
    MADRONA_NO_INLINE void setPfcPauseTimer(PortTimers &timers, Time t);
    MADRONA_NO_INLINE void setPfcResumeTimer(PortTimers &timers, Time t);
    MADRONA_NO_INLINE void clearBacklogDrainTimer(PortTimers &timers);
    MADRONA_NO_INLINE void clearPfcPauseTimer(PortTimers &timers);
    MADRONA_NO_INLINE void clearPfcResumeTimer(PortTimers &timers);
    MADRONA_NO_INLINE int32_t countActiveBacklogDrainTimers(
        madrona::Context &ctx) const;
    MADRONA_NO_INLINE int32_t countActivePfcPauseTimers(
        madrona::Context &ctx) const;
    MADRONA_NO_INLINE int32_t countActivePfcResumeTimers(
        madrona::Context &ctx) const;
    MADRONA_NO_INLINE bool hasActiveBacklogDrainTimers(
        madrona::Context &ctx) const;
    MADRONA_NO_INLINE bool hasActivePfcPauseTimers(
        madrona::Context &ctx) const;
    MADRONA_NO_INLINE bool hasActivePfcResumeTimers(
        madrona::Context &ctx) const;
    MADRONA_NO_INLINE void applyDrainHintOnePort(
        int32_t port_id,
        PortDrainHint &hint,
        PortTimers &timers);
    MADRONA_NO_INLINE void applyPfcTimerOnePort(
        int32_t ingress_port_id,
        PortPfcState &state,
        PortTimers &timers);

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
    mutable int32_t bfsDist[MAX_TOPO_NODES];
    mutable int32_t bfsQueue[MAX_TOPO_NODES];
    FlowId flowLookupBase = 0;
    int32_t flowLookupSpan = 0;
    madrona::Entity flowMetaEntityLookup[MAX_FLOWS];
    madrona::Entity flowMetaEntities[MAX_FLOWS];

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
    uint64_t systemLogStep;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
