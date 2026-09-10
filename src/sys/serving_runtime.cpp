#include "serving_runtime.hpp"

#include "kv_transfer.hpp"
#include "net_sys_interface.hpp"
#include "pd_router.hpp"
#include "serving_scheduler.hpp"

namespace madsimple {

namespace {

inline void failServing(Engine &ctx, ServingError error)
{
    SystemStatus &status = ctx.singleton<SystemStatus>();
    status.failed = 1;
    status.error_code = static_cast<int32_t>(error);
}

// Per-worker rank layout: each worker carries an independent
// (rank_start, rank_count). P and D pools no longer need equal widths.
struct WorkerSpan {
    int32_t rank_start;
    int32_t rank_count;
};

inline bool spansOverlap(const WorkerSpan &a, const WorkerSpan &b)
{
    return a.rank_start < b.rank_start + b.rank_count &&
        b.rank_start < a.rank_start + a.rank_count;
}

// Read the per-worker (rank_start, rank_count) list packed two int64 per
// worker at IC_{P,D}_WORKERS_BASE.
MADRONA_NO_INLINE bool readWorkerSpans(
    Engine &ctx, const InferenceConfigData &config, int32_t base,
    int32_t count, WorkerSpan *out)
{
    for (int32_t i = 0; i < count; i++) {
        const int64_t rank_start = config.data[base + 2 * i];
        const int64_t rank_count = config.data[base + 2 * i + 1];
        if (rank_start < 0 || rank_count <= 0 ||
            rank_count > MAX_SERVING_WORKER_RANKS ||
            rank_start + rank_count > ctx.data().numNpus) {
            return false;
        }
        out[i] = WorkerSpan {
            .rank_start = static_cast<int32_t>(rank_start),
            .rank_count = static_cast<int32_t>(rank_count),
        };
    }
    return true;
}

MADRONA_NO_INLINE bool initializeServing(
    Engine &ctx, ServingRuntime &runtime,
    const InferenceConfigData &config,
    const ServingRequestData &input)
{
    runtime = ServingRuntime {};
    runtime.enabled = config.data[IC_ENABLED] != 0;
    if (runtime.enabled == 0) {
        runtime.initialized = 1;
        return true;
    }
    const int32_t nr = static_cast<int32_t>(config.data[IC_NUM_REQUESTS]);
    const int32_t np = static_cast<int32_t>(config.data[IC_NUM_P_WORKERS]);
    const int32_t nd = static_cast<int32_t>(config.data[IC_NUM_D_WORKERS]);
    if (nr < 0 || nr > MAX_SERVING_REQUESTS ||
        np <= 0 || np > MAX_SERVING_WORKERS ||
        nd <= 0 || nd > MAX_SERVING_WORKERS ||
        config.data[IC_P_MAX_BATCH] <= 0 ||
        config.data[IC_P_MAX_BATCH] > MAX_SERVING_BATCH ||
        config.data[IC_D_MAX_BATCH] <= 0 ||
        config.data[IC_D_MAX_BATCH] > MAX_SERVING_BATCH ||
        config.data[IC_KV_PARTITION_FACTOR] < 0 ||
        (config.data[IC_KV_MODE] != 0 && config.data[IC_KV_MODE] != 1) ||
        config.data[IC_KV_LATENT_DIM] < 0 ||
        config.data[IC_WORKLOAD_PARAMS_TABLE_COUNT] < 0 ||
        config.data[IC_WORKLOAD_PARAMS_TABLE_COUNT] >
            MAX_WORKLOAD_PARAMS_TABLE_ENTRIES ||
        config.data[IC_NUM_LAYERS] <= 0 ||
        config.data[IC_NUM_KV_HEADS] <= 0 ||
        config.data[IC_HEAD_DIM] <= 0 ||
        config.data[IC_BYTES_PER_ELEM] <= 0) {
        failServing(ctx, ServingError::InvalidConfig);
        return false;
    }

    WorkerSpan p_spans[MAX_SERVING_WORKERS] {};
    WorkerSpan d_spans[MAX_SERVING_WORKERS] {};
    if (!readWorkerSpans(ctx, config, IC_P_WORKERS_BASE, np, p_spans) ||
        !readWorkerSpans(ctx, config, IC_D_WORKERS_BASE, nd, d_spans)) {
        failServing(ctx, ServingError::InvalidRankRange);
        return false;
    }

    // Check that all workers (P and D together) own disjoint rank ranges.
    for (int32_t i = 0; i < np; i++) {
        for (int32_t j = i + 1; j < np; j++) {
            if (spansOverlap(p_spans[i], p_spans[j])) {
                failServing(ctx, ServingError::InvalidRankRange);
                return false;
            }
        }
    }
    for (int32_t i = 0; i < nd; i++) {
        for (int32_t j = i + 1; j < nd; j++) {
            if (spansOverlap(d_spans[i], d_spans[j])) {
                failServing(ctx, ServingError::InvalidRankRange);
                return false;
            }
        }
    }
    for (int32_t i = 0; i < np; i++) {
        for (int32_t j = 0; j < nd; j++) {
            if (spansOverlap(p_spans[i], d_spans[j])) {
                failServing(ctx, ServingError::InvalidRankRange);
                return false;
            }
        }
    }
    // GQA same-head pairing pre-validation: every P/D worker pair must
    // satisfy the head-multiple layout rules (initializeServing runs once
    // per world; per-worker widths vary, so all pairs are checked here).
    // beginServingKvTransfer re-checks the selected pair at runtime.
    // MLA (kv_mode == 1) has no KV heads to pair: the divisibility matrix
    // is skipped entirely and only the latent width is required.
    const int64_t kv_heads = config.data[IC_NUM_KV_HEADS];
    const int64_t kv_latent_dim = config.data[IC_KV_LATENT_DIM];
    if (config.data[IC_KV_MODE] == 1) {
        if (kv_latent_dim <= 0) {
            failServing(ctx, ServingError::InvalidConfig);
            return false;
        }
    } else {
        if (kv_latent_dim != 0) {
            failServing(ctx, ServingError::InvalidConfig);
            return false;
        }
        for (int32_t i = 0; i < np; i++) {
            for (int32_t j = 0; j < nd; j++) {
                if (!gqaServingLayoutValid(
                        p_spans[i].rank_count, d_spans[j].rank_count,
                        kv_heads)) {
                    failServing(ctx, ServingError::InvalidConfig);
                    return false;
                }
            }
        }
    }
    // An explicit kv_partition_factor>0 must restate the per-rank GQA
    // layout of every P worker (== its rank_count); intermediate values
    // cannot pair heads cleanly (MLA uses kv_mode=1 and requires factor
    // ∈ {0, 1} instead — checked by beginServingKvTransfer).
    if (config.data[IC_KV_MODE] == 0 && config.data[IC_KV_PARTITION_FACTOR] > 0) {
        for (int32_t i = 0; i < np; i++) {
            if (config.data[IC_KV_PARTITION_FACTOR] !=
                    p_spans[i].rank_count) {
                failServing(ctx, ServingError::InvalidConfig);
                return false;
            }
        }
    }

    runtime.num_requests = nr;
    int64_t previous_arrival = -1;
    for (int32_t i = 0; i < nr; i++) {
        ServingRequestRecord &request = runtime.requests[i];
        request.request_id = input.data[i][SR_REQUEST_ID];
        request.arrival_ns = input.data[i][SR_ARRIVAL_NS];
        request.prompt_len = input.data[i][SR_PROMPT_LEN];
        request.output_len = input.data[i][SR_OUTPUT_LEN];
        request.state = ServingRequestState::WaitingArrival;
        if (request.arrival_ns < previous_arrival ||
            request.arrival_ns < 0 || request.prompt_len <= 0 ||
            request.output_len < 0) {
            failServing(ctx, ServingError::InvalidConfig);
            return false;
        }
        for (int32_t j = 0; j < i; j++) {
            if (runtime.requests[j].request_id == request.request_id) {
                failServing(ctx, ServingError::InvalidConfig);
                return false;
            }
        }
        previous_arrival = request.arrival_ns;
        net_sys_interface::addEvent(ctx,
            static_cast<uint64_t>(request.arrival_ns));
    }
    for (int32_t i = 0; i < np; i++) {
        runtime.p_workers[i].rank_start = p_spans[i].rank_start;
        runtime.p_workers[i].rank_count = p_spans[i].rank_count;
    }
    for (int32_t i = 0; i < nd; i++) {
        runtime.d_workers[i].rank_start = d_spans[i].rank_start;
        runtime.d_workers[i].rank_count = d_spans[i].rank_count;
    }

    // Preserve the parsed workload exactly once, then park serving ranks.
    for (int32_t i = 0; i < ctx.data().numNpus; i++) {
        const madrona::Entity entity = ctx.data().npuEntities[i];
        if (entity == madrona::Entity::none()) {
            continue;
        }
        ChakraNodes &active = ctx.get<ChakraNodes>(entity);
        ChakraTemplateNodes &templ = ctx.get<ChakraTemplateNodes>(entity);
        for (int32_t n = 0; n < MAX_CHAKRA_NODES_PER_NPU; n++) {
            templ.nodes[n] = active.nodes[n];
            active.nodes[n].type = ChakraNodeType::None;
        }
        ctx.get<OneNPUFinishedFlag>(entity).is_finished = true;
        ctx.get<ServingNpuExecution>(entity) = ServingNpuExecution {};
    }
    runtime.initialized = 1;
    return true;
}

MADRONA_NO_INLINE bool workerGenerationFinished(
    Engine &ctx, ServingWorker &worker,
    ServingStage stage, int32_t worker_id)
{
    if (worker.busy == 0) {
        return false;
    }
    for (int32_t rank = 0; rank < worker.rank_count; rank++) {
        const madrona::Entity entity =
            ctx.data().npuEntities[worker.rank_start + rank];
        if (entity == madrona::Entity::none()) {
            return false;
        }
        const ServingNpuExecution &execution =
            ctx.get<ServingNpuExecution>(entity);
        if (execution.active == 0 ||
            execution.worker_id != worker_id ||
            execution.stage != static_cast<int32_t>(stage) ||
            execution.generation != static_cast<int32_t>(worker.generation) ||
            !ctx.get<OneNPUFinishedFlag>(entity).is_finished) {
            return false;
        }
    }
    for (int32_t rank = 0; rank < worker.rank_count; rank++) {
        const madrona::Entity entity =
            ctx.data().npuEntities[worker.rank_start + rank];
        ctx.get<ServingNpuExecution>(entity).active = 0;
    }
    return true;
}

}

void servingPreUpdate(Engine &ctx)
{
    const InferenceConfigData &config =
        ctx.get<InferenceConfigData>(ctx.data().init_entity);
    ServingRuntime &runtime = ctx.singleton<ServingRuntime>();
    if (runtime.initialized == 0) {
        const ServingRequestData &input =
            ctx.get<ServingRequestData>(ctx.data().init_entity);
        if (!initializeServing(ctx, runtime, config, input)) {
            return;
        }
    }
    if (runtime.enabled == 0 || ctx.singleton<SystemStatus>().failed != 0) {
        return;
    }

    const int64_t now =
        static_cast<int64_t>(net_sys_interface::getCurrentTime(ctx));
    while (runtime.next_arrival < runtime.num_requests &&
           runtime.requests[runtime.next_arrival].arrival_ns <= now) {
        const int32_t slot = runtime.next_arrival++;
        ServingRequestRecord &request = runtime.requests[slot];
        const int32_t worker =
            selectServingWorker(runtime, config, ServingStage::Prefill);
        request.p_worker = worker;
        request.state = ServingRequestState::WaitingPrefill;
        if (worker < 0 ||
            !enqueueServingRequest(runtime.p_workers[worker], slot,
                                   request.prompt_len)) {
            failServing(ctx, ServingError::QueueOverflow);
            return;
        }
    }
    scheduleServingPrefill(ctx, runtime, config);
    scheduleServingDecode(ctx, runtime, config);
    updateServingStats(ctx);
}

void servingPostUpdate(Engine &ctx)
{
    const InferenceConfigData &config =
        ctx.get<InferenceConfigData>(ctx.data().init_entity);
    if (config.data[IC_ENABLED] == 0) {
        return;
    }
    ServingRuntime &runtime = ctx.singleton<ServingRuntime>();
    if (runtime.initialized == 0 || ctx.singleton<SystemStatus>().failed != 0) {
        return;
    }
    const int64_t now =
        static_cast<int64_t>(net_sys_interface::getCurrentTime(ctx));
    const int32_t np =
        static_cast<int32_t>(config.data[IC_NUM_P_WORKERS]);
    for (int32_t w = 0; w < np; w++) {
        ServingWorker &worker = runtime.p_workers[w];
        if (!workerGenerationFinished(ctx, worker, ServingStage::Prefill, w)) {
            continue;
        }
        worker.busy = 0;
        for (int32_t i = 0; i < worker.batch_count; i++) {
            const int32_t slot = worker.batch[i];
            runtime.requests[slot].t_p_finish_ns = now;
            beginServingKvTransfer(ctx, runtime, config, slot);
        }
        worker.batch_count = 0;
        worker.batch_tokens = 0;
    }

    const int32_t nd =
        static_cast<int32_t>(config.data[IC_NUM_D_WORKERS]);
    for (int32_t w = 0; w < nd; w++) {
        ServingWorker &worker = runtime.d_workers[w];
        if (!workerGenerationFinished(ctx, worker, ServingStage::Decode, w)) {
            continue;
        }
        worker.busy = 0;
        int32_t kept = 0;
        for (int32_t i = 0; i < worker.active_count; i++) {
            const int32_t slot = worker.active[i];
            ServingRequestRecord &request = runtime.requests[slot];
            if (request.output_done == 0) {
                request.t_first_token_ns = now;
            }
            request.output_done++;
            if (request.output_done >= request.output_len) {
                request.output_done = request.output_len;
                request.t_finish_ns = now;
                request.state = ServingRequestState::Finished;
                runtime.finished_requests++;
            } else {
                worker.active[kept++] = slot;
            }
        }
        worker.active_count = kept;
        worker.batch_count = 0;
        worker.batch_tokens = 0;
    }
    ctx.singleton<SystemStatus>().finished =
        servingIsFinished(runtime, config) ? 1 : 0;
    updateServingStats(ctx);
}

bool servingIsFinished(const ServingRuntime &runtime,
                       const InferenceConfigData &config)
{
    if (runtime.finished_requests != runtime.num_requests ||
        runtime.next_arrival != runtime.num_requests ||
        runtime.inflight_kv != 0) {
        return false;
    }
    const int32_t np =
        static_cast<int32_t>(config.data[IC_NUM_P_WORKERS]);
    const int32_t nd =
        static_cast<int32_t>(config.data[IC_NUM_D_WORKERS]);
    for (int32_t i = 0; i < np; i++) {
        const ServingWorker &w = runtime.p_workers[i];
        if (w.busy || w.queue_count || w.active_count || w.batch_count ||
            w.pending_kv_tokens) {
            return false;
        }
    }
    for (int32_t i = 0; i < nd; i++) {
        const ServingWorker &w = runtime.d_workers[i];
        if (w.busy || w.queue_count || w.active_count || w.batch_count ||
            w.pending_kv_tokens) {
            return false;
        }
    }
    return true;
}

void updateServingStats(Engine &ctx)
{
    const ServingRuntime &runtime = ctx.singleton<ServingRuntime>();
    ServingStatsData &stats = ctx.singleton<ServingStatsData>();
    for (int32_t i = 0; i < runtime.num_requests; i++) {
        const ServingRequestRecord &r = runtime.requests[i];
        int64_t *out = stats.data[i];
        out[0] = r.request_id;
        out[1] = r.arrival_ns;
        out[2] = r.t_p_start_ns;
        out[3] = r.t_p_finish_ns;
        out[4] = r.t_d_route_ns;
        out[5] = r.t_kv_start_ns;
        out[6] = r.t_kv_done_ns;
        out[7] = r.t_d_start_ns;
        out[8] = r.t_finish_ns;
        out[9] = r.p_worker;
        out[10] = r.d_worker;
        out[11] = r.output_len;
        out[12] = r.output_done;
        out[13] = r.kv_bytes;
        out[14] = static_cast<int64_t>(r.state);
        out[15] = r.prompt_len;
        out[16] = r.kv_shards;
        out[17] = r.kv_done;
        out[18] = r.t_first_token_ns;
    }
}

}
