#include "inference_scheduler.hpp"

#include "chakra_node_processing.hpp"
#include "net_sys_interface.hpp"
#include <cstdio>
#include <cstdlib>

namespace madsimple {

namespace {

inline int64_t nowNs(Engine &ctx)
{
    return static_cast<int64_t>(net_sys_interface::getCurrentTime(ctx));
}

}

// A10: the scheduling decision no longer copies the answer sheets inline.
// It appends one roster entry per (worker, rank); the per-NPU parallel
// task-graph node (inferenceResetLocal) performs the copy/reset. The
// roster is cleared at the head of every inference pre-step (see
// inferencePreUpdate), so entries never leak across frames.
void startInferenceWork(Engine &ctx, InferenceWorker &worker,
                      InferenceStage stage, int32_t worker_id,
                      int64_t token_count)
{
    worker.busy = 1;
    worker.generation++;
    worker.batch_tokens = token_count;
    InferenceResetRoster &roster = ctx.singleton<InferenceResetRoster>();
    for (int32_t rank = 0; rank < worker.rank_count; rank++) {
        const int32_t npu_id = worker.rank_start + rank;
        if (npu_id < 0 || npu_id >= ctx.data().numNpus) {
            continue;
        }
        if (roster.count >= INFERENCE_RESET_ROSTER_CAP) {
            // Structurally unreachable: entries per frame <= numNpus
            // (disjoint rank ranges enforced at init) <= 16 << 256.
            roster.overflow_count += 1;
            SystemStatus &status = ctx.singleton<SystemStatus>();
            status.failed = 1;
            status.error_code =
                static_cast<int32_t>(InferenceError::CapacityExceeded);
            return;
        }
        roster.entries[roster.count++] = InferenceResetEntry {
            .npu_id = npu_id,
            .stage = static_cast<int32_t>(stage),
            .worker_id = worker_id,
            .generation = static_cast<int32_t>(worker.generation),
            .request_count = worker.batch_count,
            .token_count = token_count,
        };
    }
}

// A10 per-NPU copy pass (see inference_scheduler.hpp). Component-reference
// version of the legacy resetNpuForReplay: identical mutations, applied
// only when this NPU appears in this frame's roster. The flow-id base is
// taken from this NPU's own id, exactly like the legacy entity lookup.
void inferenceResetLocal(Engine &ctx, uint32_t npu_id,
                         ChakraNodes &active, ChakraTemplateNodes &templ,
                         HardwareResource &hardwareResource,
                         ProcessingCompTask &processingCompTask,
                         ProcessingCommTasks &comm,
                         OneNPUFinishedFlag &oneNPUFinishedFlag,
                         ChakraNodesForNoDP &noDP,
                         ChakraReadyState &readyState,
                         InferenceNpuExecution &execution)
{
    const InferenceConfigData &config =
        ctx.get<InferenceConfigData>(ctx.data().init_entity);
    if (config.data[IC_ENABLED] == 0) {
        return;
    }
    const InferenceRuntime &runtime = ctx.singleton<InferenceRuntime>();
    if (runtime.initialized == 0 ||
        ctx.singleton<SystemStatus>().failed != 0) {
        return;
    }
    const InferenceResetRoster &roster = ctx.singleton<InferenceResetRoster>();

    for (int32_t i = 0; i < roster.count; i++) {
        const InferenceResetEntry &entry = roster.entries[i];
        if (entry.npu_id != static_cast<int32_t>(npu_id)) {
            continue;
        }

        for (int32_t n = 0; n < MAX_CHAKRA_NODES_PER_NPU; n++) {
            active.nodes[n] = templ.nodes[n];
        }
        // B1: the restored template changes the node graph, so rebuild the
        // incremental ready set for this generation (once per NPU start).
        llm_system::buildChakraReadyState(active, readyState);
        hardwareResource = HardwareResource {};
        processingCompTask = ProcessingCompTask {};
        const int64_t flow_base =
            static_cast<int64_t>(npu_id) * FLOW_ID_MAX_LENGTH;
        comm = ProcessingCommTasks {};
        comm.flow_id = flow_base;
        oneNPUFinishedFlag.is_finished = false;
        noDP = ChakraNodesForNoDP {};
        execution.active = 1;
        execution.worker_id = entry.worker_id;
        execution.stage = entry.stage;
        execution.generation = entry.generation;
        execution.request_count = entry.request_count;
        execution.token_count = entry.token_count;
        break;
    }
}

void scheduleInferencePrefill(Engine &ctx, InferenceRuntime &runtime,
                            const InferenceConfigData &config)
{
    const int32_t num_workers =
        static_cast<int32_t>(config.data[IC_NUM_P_WORKERS]);
    const int32_t max_batch = static_cast<int32_t>(
        config.data[IC_P_MAX_BATCH] > 0 ? config.data[IC_P_MAX_BATCH] : 1);
    const int64_t max_tokens = config.data[IC_P_MAX_BATCH_TOKENS];
    const int64_t max_wait = config.data[IC_P_MAX_WAIT_NS];
    const int64_t now = nowNs(ctx);
    for (int32_t w = 0; w < num_workers; w++) {
        InferenceWorker &worker = runtime.p_workers[w];
        if (worker.busy != 0 || worker.queue_count == 0) {
            continue;
        }
        // Dual-threshold start gate: launch the batch only when it is full
        // (queue can fill max_batch) or the head request has already waited
        // at least p_max_wait_ns. max_wait == 0 keeps the old
        // start-as-soon-as-idle behavior. This is a peek-only decision: the
        // queue must not be mutated when we hold this frame.
        const int64_t head_wait =
            now - runtime.requests[worker.queue[0]].arrival_ns;
        const bool batch_full = worker.queue_count >= max_batch;
        if (!(max_wait <= 0 || batch_full || head_wait >= max_wait)) {
            continue;
        }
        worker.batch_count = 0;
        int64_t tokens = 0;
        while (worker.queue_count > 0 && worker.batch_count < max_batch) {
            const int32_t slot = worker.queue[0];
            const int64_t request_tokens = runtime.requests[slot].prompt_len;
            if (worker.batch_count > 0 && max_tokens > 0 &&
                tokens + request_tokens > max_tokens) {
                break;
            }
            for (int32_t i = 1; i < worker.queue_count; i++) {
                worker.queue[i - 1] = worker.queue[i];
            }
            worker.queue_count--;
            worker.queued_tokens -= request_tokens;
            worker.batch[worker.batch_count++] = slot;
            tokens += request_tokens;
            InferenceRequestRecord &request = runtime.requests[slot];
            request.state = InferenceRequestState::Prefilling;
            request.t_p_start_ns = now;
        }
        worker.batch_id = runtime.next_batch_id++;
        startInferenceWork(ctx, worker, InferenceStage::Prefill, w, tokens);
    }
}

void scheduleInferenceDecode(Engine &ctx, InferenceRuntime &runtime,
                           const InferenceConfigData &config)
{
    const int32_t num_workers =
        static_cast<int32_t>(config.data[IC_NUM_D_WORKERS]);
    const int32_t max_batch = static_cast<int32_t>(
        config.data[IC_D_MAX_BATCH] > 0 ? config.data[IC_D_MAX_BATCH] : 1);
    const int64_t max_tokens = config.data[IC_D_MAX_BATCH_TOKENS];
    const int64_t max_wait = config.data[IC_D_MAX_WAIT_NS];
    const int64_t now = nowNs(ctx);
    for (int32_t w = 0; w < num_workers; w++) {
        InferenceWorker &worker = runtime.d_workers[w];
        if (worker.busy != 0) {
            continue;
        }
        // Dual-threshold start gate, symmetric with scheduleInferencePrefill.
        // With an empty queue the active set launches its next generation
        // right away ("有东西就跑", not gated). With a queued request the
        // generation only starts when d_max_wait_ns == 0 (plain continuous
        // batching), the batch would be full (active + queue can fill
        // max_batch, or the head request cannot join because of the token
        // cap, treating the batch as full), or the head request has already
        // waited at least d_max_wait_ns since its KV transfer finished
        // (t_kv_done_ns, Q-B decision). Peek-only decision while holding:
        // the queue must not be mutated this frame.
        if (worker.queue_count > 0) {
            const int64_t head_wait =
                now - runtime.requests[worker.queue[0]].t_kv_done_ns;
            const bool batch_full =
                worker.active_count + worker.queue_count >= max_batch;
            bool head_blocked = false;
            if (worker.active_count > 0 && max_tokens > 0) {
                int64_t active_tokens = 0;
                for (int32_t i = 0; i < worker.active_count; i++) {
                    const InferenceRequestRecord &r =
                        runtime.requests[worker.active[i]];
                    active_tokens += r.prompt_len + r.output_done;
                }
                const InferenceRequestRecord &head =
                    runtime.requests[worker.queue[0]];
                head_blocked = active_tokens + head.prompt_len +
                    head.output_done > max_tokens;
            }
            if (!(max_wait <= 0 || batch_full || head_blocked ||
                  head_wait >= max_wait)) {
                continue;
            }
        }
        // Capacity-conditioned admission at launch: pull as many queued
        // KV-ready requests as fit (active_count < d_max_batch, tokens
        // <= d_max_tokens; the first request is never token-checked), then
        // start this generation.
        int64_t tokens = 0;
        for (int32_t i = 0; i < worker.active_count; i++) {
            const InferenceRequestRecord &r =
                runtime.requests[worker.active[i]];
            tokens += r.prompt_len + r.output_done;
        }
        while (worker.queue_count > 0 && worker.active_count < max_batch) {
            const int32_t slot = worker.queue[0];
            InferenceRequestRecord &r = runtime.requests[slot];
            const int64_t context = r.prompt_len + r.output_done;
            if (worker.active_count > 0 && max_tokens > 0 &&
                tokens + context > max_tokens) {
                break;
            }
            for (int32_t i = 1; i < worker.queue_count; i++) {
                worker.queue[i - 1] = worker.queue[i];
            }
            worker.queue_count--;
            worker.queued_tokens -= context;
            worker.active[worker.active_count++] = slot;
            tokens += context;
            r.state = InferenceRequestState::Decoding;
            if (r.t_d_start_ns == 0) {
                r.t_d_start_ns = now;
            }
        }
        if (worker.active_count == 0) {
            continue;
        }
        worker.batch_count = worker.active_count;
        for (int32_t i = 0; i < worker.active_count; i++) {
            worker.batch[i] = worker.active[i];
        }
        worker.batch_id = runtime.next_batch_id++;
        startInferenceWork(ctx, worker, InferenceStage::Decode, w, tokens);
    }
}

}
