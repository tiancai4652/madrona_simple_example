#pragma once

#include <cstdint>

namespace madsimple {

constexpr int32_t MAX_INFERENCE_REQUESTS = 1024;
constexpr int32_t MAX_INFERENCE_WORKERS = 16;
constexpr int32_t MAX_INFERENCE_WORKER_RANKS = 16;
constexpr int32_t MAX_INFERENCE_QUEUE = MAX_INFERENCE_REQUESTS;
constexpr int32_t MAX_INFERENCE_BATCH = 128;
constexpr int32_t INFERENCE_CONFIG_LENGTH = 256;
constexpr int32_t INFERENCE_REQUEST_FIELDS = 4;
constexpr int32_t INFERENCE_STATS_FIELDS = 19;
constexpr int32_t MAX_WORKLOAD_PARAMS_TABLE_ENTRIES = 1024;
constexpr int32_t WORKLOAD_PARAMS_TABLE_FIELDS = 6;

// InferenceConfigData.data[] schema. All time values are nanoseconds.
enum InferenceConfigIndex : int32_t {
    IC_ENABLED = 0,
    IC_NUM_REQUESTS = 1,
    IC_NUM_P_WORKERS = 2,
    IC_NUM_D_WORKERS = 3,
    // Deprecated flat-pool slots kept so later indices never shift; the
    // per-worker lists at IC_P_WORKERS_BASE / IC_D_WORKERS_BASE replaced
    // them when workers became independently sized (rank_start/rank_count
    // per worker).
    IC_P_RANK_START = 4, // deprecated: see IC_P_WORKERS_BASE
    IC_P_RANKS_PER_WORKER = 5, // deprecated: see IC_P_WORKERS_BASE
    IC_D_RANK_START = 6, // deprecated: see IC_D_WORKERS_BASE
    IC_D_RANKS_PER_WORKER = 7, // deprecated: see IC_D_WORKERS_BASE
    IC_P_ROUTE_POLICY = 8, // deprecated: routing is fixed to QueueDepth
    IC_D_ROUTE_POLICY = 9, // deprecated: routing is fixed to QueueDepth
    IC_P_MAX_BATCH = 10,
    IC_P_MAX_BATCH_TOKENS = 11,
    IC_D_MAX_BATCH = 12,
    IC_D_MAX_BATCH_TOKENS = 13,
    IC_NUM_LAYERS = 14,
    IC_NUM_KV_HEADS = 15,
    IC_HEAD_DIM = 16,
    IC_BYTES_PER_ELEM = 17,
    IC_KV_PARTITION_FACTOR = 18,
    IC_PREFILL_REFERENCE_TOKENS = 19,
    IC_DECODE_REFERENCE_TOKENS = 20,
    IC_P_MAX_WAIT_NS = 21,
    IC_WORKLOAD_PARAMS_TABLE_COUNT = 22,
    IC_D_MAX_WAIT_NS = 23,
    // KV cache layout mode: 0 = GQA (per-head-block sharding, see
    // IC_KV_PARTITION_FACTOR), 1 = MLA (point-to-point latent replication:
    // one full-latent stream from the P pool's first rank to every rank of
    // the routed D worker, see IC_KV_LATENT_DIM).
    IC_KV_MODE = 24,
    // Merged MLA latent width per layer/token (kv_lora_rank +
    // qk_rope_head_dim, e.g. DeepSeek-V3 = 576). Must be 0 in GQA mode.
    IC_KV_LATENT_DIM = 25,
};

// Per-worker rank layout lists. Each worker occupies two int64 slots
// (rank_start, rank_count); up to MAX_INFERENCE_WORKERS (16) workers fit
// in the 32-slot window of each pool.
constexpr int32_t IC_P_WORKERS_BASE = 32;
constexpr int32_t IC_D_WORKERS_BASE = 64;
constexpr int32_t IC_WORKER_SLOTS = 2 * MAX_INFERENCE_WORKERS;

enum InferenceRequestInputIndex : int32_t {
    IR_REQUEST_ID = 0,
    IR_ARRIVAL_NS = 1,
    IR_PROMPT_LEN = 2,
    IR_OUTPUT_LEN = 3,
};

enum class InferenceStage : int32_t {
    None = 0,
    Prefill = 1,
    Decode = 2,
};

enum class InferenceRequestState : int32_t {
    Empty = 0,
    WaitingArrival,
    WaitingPrefill,
    Prefilling,
    KvTransferring,
    WaitingDecode,
    Decoding,
    Finished,
    Rejected,
};

enum class InferenceError : int32_t {
    None = 0,
    InvalidConfig = 100,
    CapacityExceeded = 101,
    InvalidRankRange = 102,
    QueueOverflow = 103,
    FlowOverflow = 104,
};

struct InferenceConfigData {
    int64_t data[INFERENCE_CONFIG_LENGTH] {};
};

struct InferenceRequestData {
    int64_t data[MAX_INFERENCE_REQUESTS][INFERENCE_REQUEST_FIELDS] {};
};

enum WorkloadParamsTableFieldIndex : int32_t {
    WPT_STAGE = 0,
    WPT_NODE_TYPE = 1,
    WPT_BATCH_SIZE = 2,
    WPT_SEQUENCE_TOKENS = 3,
    WPT_DURATION_NS = 4,
    WPT_COMM_SIZE_BYTES = 5,
};

struct WorkloadParamsTableData {
    int64_t data[MAX_WORKLOAD_PARAMS_TABLE_ENTRIES][WORKLOAD_PARAMS_TABLE_FIELDS] {};
};

struct InferenceRequestRecord {
    int64_t request_id = 0;
    int64_t arrival_ns = 0;
    int64_t prompt_len = 0;
    int64_t output_len = 0;
    int64_t output_done = 0;
    int64_t t_p_start_ns = 0;
    int64_t t_p_finish_ns = 0;
    int64_t t_d_route_ns = 0;
    int64_t t_kv_start_ns = 0;
    int64_t t_kv_done_ns = 0;
    int64_t t_d_start_ns = 0;
    int64_t t_first_token_ns = 0;
    int64_t t_finish_ns = 0;
    int64_t kv_bytes = 0;
    int32_t p_worker = -1;
    int32_t d_worker = -1;
    int32_t kv_shards = 0;
    int32_t kv_done = 0;
    uint32_t kv_done_mask = 0;
    InferenceRequestState state = InferenceRequestState::Empty;
};

struct InferenceWorker {
    int32_t rank_start = 0;
    int32_t rank_count = 0;
    int32_t queue_count = 0;
    int32_t queue[MAX_INFERENCE_QUEUE] {};
    int32_t active_count = 0;
    int32_t active[MAX_INFERENCE_BATCH] {};
    int32_t batch_count = 0;
    int32_t batch[MAX_INFERENCE_BATCH] {};
    int64_t queued_tokens = 0;
    int64_t pending_kv_tokens = 0;
    int64_t batch_tokens = 0;
    int64_t batch_id = 0;
    int64_t generation = 0;
    int32_t busy = 0;
};

struct InferenceRuntime {
    int32_t initialized = 0;
    int32_t enabled = 0;
    int32_t num_requests = 0;
    int32_t next_arrival = 0;
    int32_t finished_requests = 0;
    int32_t inflight_kv = 0;
    int64_t next_batch_id = 1;
    InferenceRequestRecord requests[MAX_INFERENCE_REQUESTS] {};
    InferenceWorker p_workers[MAX_INFERENCE_WORKERS] {};
    InferenceWorker d_workers[MAX_INFERENCE_WORKERS] {};
};

// Rows are request slots. Columns:
// id, arrival, p-start, p-finish, d-route, kv-start, kv-done, d-start,
// finish, P worker, D worker, output target, output done, kv bytes, state,
// prompt length, kv shard count, kv shard done, first-token completion.
struct InferenceStatsData {
    int64_t data[MAX_INFERENCE_REQUESTS][INFERENCE_STATS_FIELDS] {};
};

struct InferenceNpuExecution {
    int32_t active = 0;
    int32_t worker_id = -1;
    int32_t stage = static_cast<int32_t>(InferenceStage::None);
    int32_t generation = 0;
    int32_t request_count = 0;
    int64_t token_count = 0;
};

constexpr uint32_t INFERENCE_KV_FLOW_BIT = 0x80000000u;
constexpr uint32_t INFERENCE_KV_SHARD_BITS = 4u;
constexpr uint32_t INFERENCE_KV_SHARD_MASK = 0xFu;

inline uint32_t makeInferenceKvFlowID(uint32_t request_slot, uint32_t shard)
{
    return INFERENCE_KV_FLOW_BIT |
        (request_slot << INFERENCE_KV_SHARD_BITS) |
        (shard & INFERENCE_KV_SHARD_MASK);
}

inline bool isInferenceKvFlowID(uint32_t flow_id)
{
    return (flow_id & INFERENCE_KV_FLOW_BIT) != 0;
}

inline uint32_t inferenceKvRequestSlot(uint32_t flow_id)
{
    return (flow_id & ~INFERENCE_KV_FLOW_BIT) >> INFERENCE_KV_SHARD_BITS;
}

inline uint32_t inferenceKvShard(uint32_t flow_id)
{
    return flow_id & INFERENCE_KV_SHARD_MASK;
}

}
