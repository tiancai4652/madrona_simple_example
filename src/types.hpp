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

struct SimDriverArch : public madrona::Archetype<
    SimDriver
> {};

using Time = double;
using Bw = double;
using Bytes = double;
using FlowId = int64_t;
using NodeId = int32_t;

enum class NodeType : int32_t {
    Host,
    Switch,
};

constexpr int32_t PFC_MAX_PRIORITY = 8;
constexpr int32_t MAX_CHUNK_WEIGHTS = 16;
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
};

// --- Per-port scratch / hint components introduced in phase B. ---
// All are plain-old-data so they are safe to live on a Port archetype that
// a ParallelForNode can iterate. Cross-port reductions, deferred entity
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

constexpr int32_t MAX_PORT_CLEANUP = 32;

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
    // Snapshot of DirtyPort captured at alloc time (before emit/clear run);
    // consumed by the cachedNextDrainTime reduction singleton. Populated by
    // allocOnePort once phase B.2 lands.
    int32_t was_dirty_at_alloc = 0;
    // Snapshot of DirtyPort captured by clearDirtyOnePort (after emit);
    // consumed by snapshotDirtyPorts to rebuild lastDirtyPortIDs in a
    // deterministic port_id ascending order.
    int32_t was_dirty_at_clear = 0;
};

struct Port : public madrona::Archetype<
    DirtyPort,
    PortState,
    PortBuffer,
    PortPfcConfig,
    PortPfcState,
    PortCachedHints,
    PortDrainHint,
    PortCleanup,
    PortTraceLast
> {};

struct FlowTag : public madrona::Archetype<
    FlowTagState
> {};

}
