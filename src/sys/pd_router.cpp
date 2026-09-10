#include "pd_router.hpp"

#include <climits>

namespace madsimple {

// Routing is fixed to queue-depth balancing: pick the worker with the
// smallest queue-depth value = total tokens waiting on that worker:
//   P side: queued_tokens + (busy ? batch_tokens + 1 : 0)
//   D side: queued_tokens + pending_kv_tokens
//           + sum(active[i].prompt_len + output_done)
// The scan runs in ascending worker id and only a *strictly smaller* value
// replaces the incumbent, so equal values keep the earliest (smallest id)
// worker: explicit tie-break by worker id.
int32_t selectServingWorker(ServingRuntime &runtime,
                            const InferenceConfigData &config,
                            ServingStage stage)
{
    const bool prefill = stage == ServingStage::Prefill;
    const int32_t count = static_cast<int32_t>(config.data[
        prefill ? IC_NUM_P_WORKERS : IC_NUM_D_WORKERS]);
    if (count <= 0) {
        return -1;
    }

    ServingWorker *workers = prefill ? runtime.p_workers : runtime.d_workers;
    int32_t selected = 0;
    int64_t best = INT64_MAX;
    for (int32_t i = 0; i < count; i++) {
        const ServingWorker &worker = workers[i];
        int64_t value = worker.queued_tokens;
        if (prefill) {
            if (worker.busy != 0) {
                value += worker.batch_tokens + 1;
            }
        } else {
            value += worker.pending_kv_tokens;
            for (int32_t active_idx = 0;
                 active_idx < worker.active_count; active_idx++) {
                const ServingRequestRecord &request =
                    runtime.requests[worker.active[active_idx]];
                value += request.prompt_len + request.output_done;
            }
        }
        if (value < best) {
            best = value;
            selected = i;
        }
    }
    return selected;
}

bool enqueueServingRequest(ServingWorker &worker, int32_t slot,
                           int64_t tokens)
{
    if (worker.queue_count >= MAX_SERVING_QUEUE) {
        return false;
    }
    worker.queue[worker.queue_count++] = slot;
    worker.queued_tokens += tokens;
    return true;
}

}
