#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE void startServingWork(
    Engine &ctx, ServingWorker &worker, ServingStage stage,
    int32_t worker_id, int64_t token_count);
MADRONA_NO_INLINE void scheduleServingPrefill(
    Engine &ctx, ServingRuntime &runtime, const InferenceConfigData &config);
MADRONA_NO_INLINE void scheduleServingDecode(
    Engine &ctx, ServingRuntime &runtime, const InferenceConfigData &config);

}
