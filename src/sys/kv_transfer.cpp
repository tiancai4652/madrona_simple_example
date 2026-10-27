#include "kv_transfer.hpp"

#include "net_sys_interface.hpp"
#include "pd_router.hpp"

#include <climits>
#include <cstdio>
#include <cstdlib>

namespace madsimple {

namespace {

inline void failInference(Engine &ctx, InferenceError error)
{
    SystemStatus &status = ctx.singleton<SystemStatus>();
    status.failed = 1;
    status.error_code = static_cast<int32_t>(error);
}

inline bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || (rhs != 0 && lhs > INT64_MAX / rhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

// GQA same-head pairing: every P/D worker width must form an integer
// multiple relationship with num_kv_heads, and the two widths must form
// one with each other, so each P rank's contiguous head block pairs with
// exactly one D rank (group).
inline bool gqaLayoutValid(int64_t p_width, int64_t d_width,
                           int64_t kv_heads)
{
    return kv_heads > 0 && p_width > 0 && d_width > 0 &&
        (kv_heads % p_width == 0 || p_width % kv_heads == 0) &&
        (kv_heads % d_width == 0 || d_width % kv_heads == 0) &&
        (d_width % p_width == 0 || p_width % d_width == 0);
}

}

bool gqaInferenceLayoutValid(int64_t p_width, int64_t d_width,
                           int64_t kv_heads)
{
    return gqaLayoutValid(p_width, d_width, kv_heads);
}

bool beginInferenceKvTransfer(Engine &ctx, InferenceRuntime &runtime,
                            const InferenceConfigData &config, int32_t slot)
{
    InferenceRequestRecord &request = runtime.requests[slot];
    const int32_t d_worker =
        selectInferenceWorker(runtime, config, InferenceStage::Decode);
    if (d_worker < 0) {
        failInference(ctx, InferenceError::InvalidConfig);
        return false;
    }
    request.d_worker = d_worker;
    runtime.d_workers[d_worker].pending_kv_tokens += request.prompt_len;
    request.t_d_route_ns =
        static_cast<int64_t>(net_sys_interface::getCurrentTime(ctx));
    request.t_kv_start_ns = request.t_d_route_ns;

    const int64_t layers = config.data[IC_NUM_LAYERS];
    const int64_t kv_heads = config.data[IC_NUM_KV_HEADS];
    const int64_t head_dim = config.data[IC_HEAD_DIM];
    const int64_t elem_bytes = config.data[IC_BYTES_PER_ELEM];
    const int64_t kv_mode = config.data[IC_KV_MODE];
    const int64_t kv_latent_dim = config.data[IC_KV_LATENT_DIM];

    const InferenceWorker &p = runtime.p_workers[request.p_worker];
    const InferenceWorker &d = runtime.d_workers[d_worker];

    if (kv_mode == 1) {
        // MLA layout: the KV cache is a single compressed latent vector
        // per layer/token shared by every TP rank (TP splits the Q
        // heads/projections, not the latent). There is no head block to
        // shard, so the GQA divisibility matrix does not apply; instead
        // the latent width must be configured and factor must be a
        // no-op (0 = default) or an explicit restatement of the single
        // shard (1).
        const int64_t kv_factor = config.data[IC_KV_PARTITION_FACTOR];
        if (kv_latent_dim <= 0 || kv_factor < 0 || kv_factor > 1) {
            failInference(ctx, InferenceError::InvalidConfig);
            return false;
        }
        int64_t latent_bytes = layers;
        if (!checkedMultiply(latent_bytes, request.prompt_len,
                             latent_bytes) ||
            !checkedMultiply(latent_bytes, kv_latent_dim, latent_bytes) ||
            !checkedMultiply(latent_bytes, elem_bytes, latent_bytes)) {
            failInference(ctx, InferenceError::CapacityExceeded);
            return false;
        }
        request.kv_bytes = latent_bytes;

        // 022-KV单播 Form-C (DP128语义, 用户已拍板): 每请求只发 1 份全量
        // latent, 由算出它的 P rank 直达其 assigned D rank (slot 取余打散,
        // 无单点)。Form-B 每请求 128 份全复制 (16K 流/batch) 在 DP 下 127 份
        // 无人需要 (仿真不建模扔, 白占带宽+堆 D-burst); DP 每个 rank 只算
        // 自己的请求, 谁算谁发, 天然一对一。KV 总流量降 128 倍。
        // 安全依据沿 Form-B: 投递只认(slot,shard), KV_BIT 命名空间隔离,
        // mask 按位记账 (现只需 bit0)。与 Form-B 轨迹不再可比 (属已授权语义变更)。
        const int32_t shards = 1;
        if (p.rank_count <= 0 || d.rank_count <= 0) {
            failInference(ctx, InferenceError::InvalidConfig);
            return false;
        }
        request.kv_shards = shards;
        request.kv_done = 0;
        request.kv_done_mask[0] = 0;
        request.kv_done_mask[1] = 0;
        request.state = InferenceRequestState::KvTransferring;
        runtime.inflight_kv += shards;
        {
            const int64_t src_rank =
                (int64_t)p.rank_start + (slot % p.rank_count);
            const int64_t dst_rank =
                (int64_t)d.rank_start + (slot % d.rank_count);
            net_sys_interface::setFlow(ctx,
                static_cast<uint32_t>(src_rank),
                static_cast<uint64_t>(src_rank),
                static_cast<uint64_t>(dst_rank),
                latent_bytes > 0 ? static_cast<uint64_t>(latent_bytes) : 1,
                makeInferenceKvFlowID(static_cast<uint32_t>(slot), 0),
                0);
        }
        return true;
    }

    // Runtime fallback for the GQA head-layout checks (initializeInference
    // pre-validates every P/D worker pair at startup).
    if (!gqaLayoutValid(p.rank_count, d.rank_count, kv_heads)) {
        failInference(ctx, InferenceError::InvalidConfig);
        return false;
    }

    // Per-rank head share: normally kv_heads / p_width heads per P rank;
    // when the P pool is wider than num_kv_heads each rank holds a
    // single (replicated) head instead.
    int64_t heads_per_p_rank = kv_heads / p.rank_count;
    if (heads_per_p_rank <= 0) {
        heads_per_p_rank = 1;
    }
    int64_t kv_bytes = layers;
    if (!checkedMultiply(kv_bytes, request.prompt_len, kv_bytes) ||
        !checkedMultiply(kv_bytes, 2, kv_bytes) ||
        !checkedMultiply(kv_bytes, heads_per_p_rank, kv_bytes) ||
        !checkedMultiply(kv_bytes, head_dim, kv_bytes) ||
        !checkedMultiply(kv_bytes, elem_bytes, kv_bytes)) {
        failInference(ctx, InferenceError::CapacityExceeded);
        return false;
    }
    request.kv_bytes = kv_bytes;

    // Shard count follows the real KV layout: each shard is the KV of the
    // contiguous head block its P rank holds. Ranks replicated onto the
    // same head (p_width > num_kv_heads, copy factor r = p_width/kv_heads)
    // stay silent and only the first rank of every copy group emits a
    // stream. An explicit kv_partition_factor>0 must restate the per-rank
    // GQA layout (== p.rank_count) and is otherwise rejected as
    // InvalidConfig: no intermediate shard count pairs heads cleanly
    // (MLA configures kv_mode=1 instead and skips this branch).
    const int64_t kv_factor = config.data[IC_KV_PARTITION_FACTOR];
    if (kv_factor > 0 && kv_factor != p.rank_count) {
        failInference(ctx, InferenceError::InvalidConfig);
        return false;
    }
    int32_t shards = static_cast<int32_t>(
        heads_per_p_rank > 1 ? p.rank_count : kv_heads);
    if (shards <= 0 || shards > p.rank_count ||
        shards > MAX_INFERENCE_WORKER_RANKS) {
        failInference(ctx, InferenceError::InvalidConfig);
        return false;
    }

    request.kv_shards = shards;
    request.kv_done = 0;
    request.kv_done_mask[0] = 0;
    request.kv_done_mask[1] = 0;
    request.state = InferenceRequestState::KvTransferring;
    runtime.inflight_kv += shards;

    // Copy factor between consecutive emitting P ranks (1 = no
    // replication, every P rank streams its own head block).
    const int64_t copy_factor =
        p.rank_count > kv_heads ? p.rank_count / kv_heads : 1;

    for (int32_t shard = 0; shard < shards; shard++) {
        const uint64_t bytes = static_cast<uint64_t>(kv_bytes);
        // Same-head pairing via the head index: the sending P rank holds
        // the contiguous head block starting at first_head, and the D rank
        // owning head h is d.rank_start + h × d_width / kv_heads. Without
        // replication shard s is the s-th P rank holding heads
        // [s×heads_per_p_rank, ...); with replication (p_width >
        // kv_heads, copy factor r = p_width/kv_heads) the s-th emitting
        // rank is rank s×r holding head s. Trailing D ranks stay idle
        // when the D pool is wider than num_kv_heads.
        const int64_t p_rank_index = shard * copy_factor;
        const int64_t first_head =
            copy_factor > 1 ? shard : shard * heads_per_p_rank;
        net_sys_interface::setFlow(ctx,
            static_cast<uint32_t>(p.rank_start + p_rank_index),
            static_cast<uint64_t>(p.rank_start + p_rank_index),
            static_cast<uint64_t>(
                d.rank_start + first_head * d.rank_count / kv_heads),
            bytes > 0 ? bytes : 1,
            makeInferenceKvFlowID(static_cast<uint32_t>(slot),
                                static_cast<uint32_t>(shard)),
            0);
    }
    return true;
}

// A1 draft pass (per NPU): see kv_transfer.hpp. Only entry-local filters
// run here; the request-state checks and all ledger mutations live in the
// commit pass, replayed in npu-major order.
void collectInferenceKvDrafts(Engine &ctx, uint32_t npu_id,
                              const NpuFlowFinishedList &finished,
                              NpuKvCollectDraft &draft)
{
    (void)npu_id;
    const InferenceConfigData &config =
        ctx.get<InferenceConfigData>(ctx.data().init_entity);
    if (config.data[IC_ENABLED] == 0) {
        return;
    }
    const InferenceRuntime &runtime = ctx.singleton<InferenceRuntime>();
    if (runtime.initialized == 0) {
        return;
    }

    for (uint32_t i = 0; i < finished.count; i++) {
        const uint32_t flow_id = finished.flows[i].flow_id;
        if (!isInferenceKvFlowID(flow_id)) {
            continue;
        }
        const uint32_t slot = inferenceKvRequestSlot(flow_id);
        const uint32_t shard = inferenceKvShard(flow_id);
        if (slot >= static_cast<uint32_t>(runtime.num_requests) ||
            shard >= MAX_INFERENCE_WORKER_RANKS) {
            continue;
        }
        if (draft.count >= MAX_FLOWS_PER_NPU) {
            // Structurally unreachable (draft entries <= finished.count
            // <= MAX_FLOWS_PER_NPU), kept as the agreed loud failure.
            draft.overflow_count += 1;
            failInference(ctx, InferenceError::QueueOverflow);
            return;
        }
        draft.tickets[draft.count++] = NpuKvCollectTicket {
            .slot = slot,
            .shard = shard,
            .end_time_ns = finished.flows[i].end_time_ns,
        };
    }
}

// A1 commit pass (serial singleton): replays the draft tickets in
// npu_id-major order with the exact legacy per-entry body, so the
// request-ledger mutations and the D-worker queue enqueue order are
// byte-identical to the pre-split singleton (F6/F7 invariants).
void commitInferenceKvCollect(Engine &ctx)
{
    const InferenceConfigData &config =
        ctx.get<InferenceConfigData>(ctx.data().init_entity);
    if (config.data[IC_ENABLED] == 0) {
        return;
    }
    InferenceRuntime &runtime = ctx.singleton<InferenceRuntime>();
    if (runtime.initialized == 0) {
        return;
    }

    for (int32_t npu_id = 0; npu_id < ctx.data().numNpus; npu_id++) {
        const madrona::Entity entity = ctx.data().npuEntities[npu_id];
        if (entity == madrona::Entity::none()) {
            continue;
        }
        NpuKvCollectDraft &draft = ctx.get<NpuKvCollectDraft>(entity);
        for (uint32_t i = 0; i < draft.count; i++) {
            const NpuKvCollectTicket &ticket = draft.tickets[i];
            const uint32_t slot = ticket.slot;
            const uint32_t shard = ticket.shard;
            InferenceRequestRecord &request = runtime.requests[slot];
            const uint32_t word = shard >> 6;
            const uint64_t bit = 1ULL << (shard & 63);
            if (request.state != InferenceRequestState::KvTransferring ||
                (request.kv_done_mask[word] & bit) != 0) {
                continue;
            }
            request.kv_done_mask[word] |= bit;
            request.kv_done++;
            const int64_t completion_ns =
                static_cast<int64_t>(ticket.end_time_ns);
            if (completion_ns > request.t_kv_done_ns) {
                request.t_kv_done_ns = completion_ns;
            }
            runtime.inflight_kv--;
            if (request.kv_done != request.kv_shards) {
                continue;
            }
            InferenceWorker &worker = runtime.d_workers[request.d_worker];
            worker.pending_kv_tokens -= request.prompt_len;
            if (worker.pending_kv_tokens < 0) {
                worker.pending_kv_tokens = 0;
            }
            if (request.output_len == 0) {
                request.t_d_start_ns = request.t_kv_done_ns;
                request.t_first_token_ns = request.t_kv_done_ns;
                request.t_finish_ns = request.t_kv_done_ns;
                request.state = InferenceRequestState::Finished;
                runtime.finished_requests++;
                continue;
            }
            request.state = InferenceRequestState::WaitingDecode;
            ctx.singleton<SimRuntimeState>().inferenceHandoffPending = 1;
            if (!enqueueInferenceRequest(worker, static_cast<int32_t>(slot),
                                       request.prompt_len)) {
                failInference(ctx, InferenceError::QueueOverflow);
            }
        }
        draft.count = 0;
    }
}

}
