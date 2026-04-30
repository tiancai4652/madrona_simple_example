#pragma once

#include <cstdint>

#include <madrona/components.hpp>

namespace madsimple {

enum class ExportID : uint32_t {
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    // GPU-mode introspection mirrors: a final per-step system copies the
    // scalar counts / simulation time and the flowCompletions[] array
    // out of the Sim struct into these singleton components on
    // SimDriverArch so mgr.cpp's GPUImpl can cudaMemcpy them back to
    // host. On CPU these are unused (CPUImpl reads Sim fields directly
    // via TaskGraphExecutor::getWorldData), but we still populate them
    // so the export layout stays identical between backends.
    SimStats,
    StepWorkloadStats,
    FlowCompletionBuf,
    NumExports,
};

struct Reset {
    int32_t resetNow;
};

enum class Action : int32_t {
    Up    = 0,
    Down  = 1,
    Left  = 2,
    Right = 3,
    None,
};

struct GridPos {
    int32_t y;
    int32_t x;
};

struct Reward {
    float r;
};

struct Done {
    float episodeDone;
};

struct CurStep {
    uint32_t step;
};

struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep
> {};

// Singleton driver used by task-graph-level systems that need a single
// parallel entry point (scheduling, event delivery, global reductions,
// singleton flush nodes). per-entity parallel systems must not live on
// SimDriverArch; they should iterate over Port / FlowTag directly.
struct SimDriver {
    int32_t tick = 0;
};

using Time = double;
using Bw = double;
using Bytes = double;
using FlowId = int64_t;
using NodeId = int32_t;

// Size of the flow-completion record history we export to the host.
// Matches Sim::flowCompletions[] capacity in sim.hpp.
constexpr int32_t MAX_FLOW_COMPLETIONS = 68608;

// Matches the layout of the completion record snapshotted by
// Sim::recordFlowCompletion(). Kept POD so it can be copied into the
// FlowCompletionBuf component that GPUImpl DMAs back to host every
// step.
struct FlowCompletionRecord {
    FlowId flow_id = -1;
    NodeId src_node = -1;
    NodeId dst_node = -1;
    Bytes size = 0.0;
    Time start_time = 0.0;
    Time end_time = 0.0;
    int32_t priority = 0;
    int32_t _pad = 0;

    inline Time fct() const
    {
        return end_time - start_time;
    }
};

// Singleton snapshot of Sim scalar counts + simulationTime. Attached to
// SimDriverArch and refreshed at the end of every step by
// updateSimStatsStepSystem. mgr.cpp's GPUImpl reads it via the exported
// column pointer + cudaMemcpy so Python-visible getters work on GPU
// without needing a public Sim pointer out of MWCudaExecutor.
struct SimStats {
    double simulationTime = 0.0;
    int32_t numFlowDefs = 0;
    int32_t numPendingFlows = 0;
    int32_t numDelayedEvents = 0;
    int32_t numActiveTags = 0;
    int32_t numSourceTags = 0;
    int32_t numFlowCompletions = 0;
    // [step-trace] driver.tick at the time this mirror was written. If
    // host sees this stuck at 0 across many steps, the task graph never
    // ran scheduleStepSystem (driver.tick++) AND/OR never ran
    // updateSimStatsStepSystem. Used to distinguish "task graph silent"
    // from "HostPrint output lost".
    int32_t lastTick = 0;
};

// Per-step workload snapshot used to judge whether the simulation exposes
// enough active work to benefit from deeper ECS / GPU parallelization.
struct StepWorkloadStats {
    int32_t step = 0;
    double now = 0.0;

    int32_t ready_flows = 0;
    int32_t scheduled_events = 0;
    int32_t pending_flows_after = 0;

    int32_t due_events = 0;
    int32_t due_arrival = 0;
    int32_t due_bwupdate = 0;
    int32_t due_pfc = 0;
    int32_t due_ports = 0;
    int32_t delayed_events_after = 0;

    int32_t dirty_ports = 0;
    int32_t last_dirty_ports = 0;
    int32_t alloc_ports = 0;
    int32_t alloc_tags = 0;
    int32_t emit_ports = 0;
    int32_t buffer_ports = 0;

    int32_t create_reqs = 0;
    int32_t create_ports = 0;
    int32_t cleanup_reqs = 0;
    int32_t cleanup_ports = 0;
    int32_t completion_reqs = 0;
    int32_t completion_ports = 0;
    int32_t outbox_events = 0;
    int32_t outbox_ports = 0;

    int32_t active_tags = 0;
    int32_t source_tags = 0;
    int32_t ingress_tags = 0;
    int32_t finished_sources = 0;
    int32_t emitted_cleanup = 0;
    int32_t drain_timers = 0;
    int32_t pause_timers = 0;
    int32_t resume_timers = 0;
    double next_dt = 0.0;
};

// Singleton mirror of Sim::flowCompletions[].record for GPU export. Same
// fixed capacity as MAX_FLOW_COMPLETIONS; entries past numFlowCompletions
// are zero-initialised and should not be read.
struct FlowCompletionBuf {
    FlowCompletionRecord records[MAX_FLOW_COMPLETIONS] {};
};

struct SimDriverArch : public madrona::Archetype<
    SimDriver
> {};

// Event structs (moved from sim.hpp so PortOutbox can embed them in the
// Port archetype). Kept POD so the Port component array stays fixed-size
// and GPU-friendly.
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

enum class NodeType : int32_t {
    Host,
    Switch,
};

constexpr int32_t PFC_MAX_PRIORITY = 8;
// Must be >= the maximum number of flows per port so that buffer chunks can
// record all active flows. leafspine1024 with d=64 has 64 flows per inter-
// switch port; use 64 to cover this topology without silently capping chunk
// weights.  If future topologies have more flows per port, raise this value
// to avoid incorrect buffer bandwidth allocation.
constexpr int32_t MAX_CHUNK_WEIGHTS = 64;
constexpr int32_t MAX_BUFFER_CHUNKS = 16;
constexpr int32_t MAX_PAUSED_UPSTREAMS = 16;

struct DirtyPort {
    int32_t isDirty = 0;
};

struct PortState {
    int32_t port_id = -1;
    NodeId node_id = -1;
    int32_t port_idx = -1;
    Bw port_bw = 0.0;
    int32_t connected = 0;
    int32_t next_port_id = -1;
};

struct FlowTagState {
    int32_t port_id = -1;
    FlowId flow_id = -1;
    int32_t priority = 0;
    Bw in_bw = 0.0;
    Bw out_bw = 0.0;
    Bw prev_out_bw = 0.0;
    Bytes backlog = 0.0;
    Time last_backlog_time = 0.0;
    Bytes remaining = 0.0;
    Time last_remaining_time = 0.0;
    int32_t is_source = 0;
    int32_t downstream_created = 0;
    int32_t next_port_id = -1;
    int32_t ingress_port_id = -1;
    // Phase D: cached owning port entity so destroyTag can remove this tag
    // from PortTagList without rescanning portEntities[]. Written by
    // createTagOnPort, cleared on tag destruction.
    madrona::Entity port_entity = madrona::Entity::none();
};

struct FlowWeight {
    FlowId flow_id = -1;
    double weight = 0.0;
};

struct BufferChunk {
    Bytes chunk_bytes = 0.0;
    int32_t num_weights = 0;
    FlowWeight weights[MAX_CHUNK_WEIGHTS] {};
};

struct PriorityBuffer {
    Bytes buf_cnt = 0.0;
    int32_t num_chunks = 0;
    int32_t head = 0;
    int32_t tail = 0;
    BufferChunk buf_chunks[MAX_BUFFER_CHUNKS] {};
    double net_buffer_rate = 0.0;
};

struct PortBuffer {
    PriorityBuffer prior_bufs[PFC_MAX_PRIORITY] {};
    Time last_update_time = 0.0;
};

struct PortPfcConfig {
    double xoff[PFC_MAX_PRIORITY] {};
    double xon[PFC_MAX_PRIORITY] {};
    int32_t pfc_enabled = 0;
};

struct PortPfcState {
    int32_t paused[PFC_MAX_PRIORITY] {};
    int32_t pause_active[PFC_MAX_PRIORITY] {};
    int32_t paused_upstream_count[PFC_MAX_PRIORITY] {};
    int32_t paused_upstreams[PFC_MAX_PRIORITY][MAX_PAUSED_UPSTREAMS] {};
    int32_t pfc_cnt[PFC_MAX_PRIORITY] {};
    // Phase C: deferred PFC timer mutations. The per-Port pfcDetectOnePort
    // sets want_* flags on its own PortPfcState; the flushPortPfcTimers
    // singleton applies clear-then-set to Sim::pfc{Pause,Resume}Timers in
    // port_id ascending order so the global timer arrays stay race-free
    // on GPU and deterministic across backends.
    int32_t want_clear_pause = 0;
    int32_t want_clear_resume = 0;
    int32_t want_set_pause = 0;
    int32_t want_set_resume = 0;
    double set_pause_t = 0.0;
    double set_resume_t = 0.0;
};

// --- Per-port scratch / hint state introduced in phase B. ---
// These remain POD so they can live either on the Port archetype or in
// Sim-owned port_id indexed arrays. Cross-port reductions, deferred entity
// destruction, and deferred log emission consume these fields in a
// deterministic singleton pass driven by SimDriverArch.

struct PortCachedHints {
    int32_t has_drain_hint = 0;
    int32_t has_finish_hint = 0;
    double drain_hint_t = 0.0;
    double finish_hint_t = 0.0;
};

struct PortDrainHint {
    int32_t want_clear = 0;
    int32_t want_set = 0;
    double set_t = 0.0;
};

// leafspine1024 d64 all-to-all drives 63 flows into each destination host
// port, so a single frame can queue >32 deferred destroyTag requests on one
// port. Keep headroom above that to avoid silently stranding tags.
constexpr int32_t MAX_PORT_CLEANUP = 64;

struct PortCleanup {
    int32_t num = 0;
    madrona::Entity tags[MAX_PORT_CLEANUP] {};
    int32_t propagate[MAX_PORT_CLEANUP] {};
};

struct PortTraceLast {
    // alloc summary (populated only when this port had at least one live tag).
    int32_t has_alloc_trace = 0;
    double alloc_port_bw = 0.0;
    int32_t alloc_num_tags = 0;
    int32_t alloc_num_live = 0;
    double alloc_sum_in = 0.0;
    double alloc_sum_out = 0.0;
    int32_t alloc_is_dest_only = 0;
    int32_t alloc_has_buffer = 0;
    // buffer summary.
    int32_t buffer_processed = 0;
    int32_t buffer_destroy_count = 0;
    double buffer_total_buf_cnt = 0.0;
    // Phase C: per-Port emit (downstream) summary.
    int32_t emit_is_dirty = 0;
    int32_t emit_arrival_count = 0;
    int32_t emit_bwupdate_count = 0;
    // Phase C: per-Port pfcDetect summary. emitted_pfc counts events pushed
    // into PortOutbox; checked counts ports that actually ran the detect
    // body (i.e. participated in the emit_pfc summary "checked_port_count").
    int32_t pfc_detect_checked = 0;
    int32_t pfc_detect_emitted = 0;
    // Snapshot of DirtyPort captured at alloc time (before emit/clear run);
    // consumed by the cachedNextDrainTime reduction singleton. Populated by
    // allocOnePort once phase B.2 lands.
    int32_t was_dirty_at_alloc = 0;
    // Snapshot of DirtyPort captured by clearDirtyOnePort (after emit);
    // consumed by snapshotDirtyPorts to rebuild lastDirtyPortIDs in a
    // deterministic port_id ascending order.
    int32_t was_dirty_at_clear = 0;
    // Phase E: ingress_chain per-Port summary counters. The per-Port
    // arrival/bwUpdate/pfcPropagate workers increment these; the
    // logIngressChainSystem singleton emits the single summary line
    // aggregated across all ports (same format as the legacy
    // printSystemArrivalSummary / printSystemBwUpdateSummary /
    // printSystemPfcSummary). The per-event detail log lines (create /
    // update / destroy / buffered_zero / forward / complete) are still
    // emitted from within the per-Port worker; scope "ingress_chain" is
    // canonical-sorted, so per-port interleaving is tolerated.
    int32_t arrival_created = 0;
    int32_t arrival_updated = 0;
    int32_t arrival_skipped = 0;
    int32_t bwupd_created = 0;
    int32_t bwupd_updated = 0;
    int32_t bwupd_buffered_zero = 0;
    int32_t bwupd_destroyed = 0;
    int32_t bwupd_forwarded = 0;
    int32_t bwupd_completed = 0;
    int32_t bwupd_skipped = 0;
    int32_t pfc_applied = 0;
    int32_t pfc_skipped = 0;
};

// Phase C: per-Port outbox for DelayedEvents produced by pfcDetectOnePort
// and emitOnePort. The per-Port worker fills `events[0..num_events)` on
// the port's own component and the singleton `flushPortOutbox` appends
// them to Sim::delayedEvents in port_id ascending order. Size is set so
// one port can hold a full pause-all-priorities wave (PFC_MAX_PRIORITY *
// MAX_PAUSED_UPSTREAMS = 128 events) plus a generous emit batch.
constexpr int32_t MAX_PORT_OUTBOX = 256;

struct PortOutbox {
    int32_t num_events = 0;
    DelayedEvent events[MAX_PORT_OUTBOX] {};
};

// Phase D: per-Port list of tag entities owned by this port (i.e. tags
// whose FlowTagState.port_id == this port). Replaces the legacy
// "for (i = 0; i < numTagIndexEntries; i++) if (tagIndex[i].port_id !=
// port_id) continue;" O(N_tags * N_ports) scan in per-Port workers.
// The Sim-level tagIndex[] is kept as a read-only mirror for findTag()
// fallback lookups until phase F.
constexpr int32_t MAX_TAGS_PER_PORT = 256;

struct PortTagList {
    int32_t count = 0;
    madrona::Entity tags[MAX_TAGS_PER_PORT] {};
};

// Per-ingress-port mirror of the non-source ingressTags[] subset. Most
// ingress-scoped systems only care about one ingress port at a time; this
// list lets them avoid rescanning the full global ingressTags[] array on the
// common path. If a port ever exceeds the fixed local capacity, overflow is
// latched and callers fall back to the authoritative global ingressTags[].
constexpr int32_t MAX_TAGS_PER_INGRESS = 256;

struct IngressTagList {
    int32_t count = 0;
    int32_t overflow = 0;
    madrona::Entity tags[MAX_TAGS_PER_INGRESS] {};
};

// Phase E: per-Port inbox. The deliverEvents singleton dispatches each
// delayedEvent whose t <= now to the target port's inbox (arrival/bwupd
// target ev.port_id, pfc target ev.target_port_id). Per-Port workers
// arrival/bwUpdate/pfcPropagate then consume their local inbox without
// scanning the global Sim::inbox* arrays.
constexpr int32_t MAX_PORT_INBOX_ARRIVAL = 64;
constexpr int32_t MAX_PORT_INBOX_BWUPD = 64;
constexpr int32_t MAX_PORT_INBOX_PFC = 32;

struct PortInbox {
    int32_t num_arrival = 0;
    FlowArrivalEv arrivals[MAX_PORT_INBOX_ARRIVAL] {};
    int32_t num_bwupd = 0;
    BwUpdateEv bwupds[MAX_PORT_INBOX_BWUPD] {};
    int32_t num_pfc = 0;
    PfcControlEv pfcs[MAX_PORT_INBOX_PFC] {};
};


// they push requests here and flushTagCreateSystem (singleton) walks
// port_id ascending and actually creates the tags via createTagOnPort.
struct PortCreateReq {
    int32_t from_arrival = 0;
    FlowId flow_id = -1;
    Bw in_bw = 0.0;
    Bytes size = 0.0;
    int32_t is_source = 0;
    int32_t priority = 0;
    int32_t log_enabled = 0;
    const char *log_label = nullptr;
};

constexpr int32_t MAX_PORT_CREATE = 64;

struct PortCreateList {
    int32_t num = 0;
    PortCreateReq reqs[MAX_PORT_CREATE] {};
};

// Phase E: deferred recordFlowCompletion requests. bwUpdate on a
// terminal port must not mutate Sim::flowCompletions / flowDefs /
// flowRoutes directly; it pushes the flow_id here and
// flushFlowCompletionSystem applies them in port_id ascending order.
// Completion notifications can also fan in at destination host ports at the
// same 63-flow scale; a 32-entry queue silently drops completions.
constexpr int32_t MAX_PORT_COMPLETE = 64;

struct PortCompletionList {
    int32_t num = 0;
    FlowId flow_ids[MAX_PORT_COMPLETE] {};
};

struct Port : public madrona::Archetype<
    PortState,
    PortBuffer
> {};

struct FlowTag : public madrona::Archetype<
    FlowTagState
> {};

}
