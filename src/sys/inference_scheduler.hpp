#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE void startInferenceWork(
    Engine &ctx, InferenceWorker &worker, InferenceStage stage,
    int32_t worker_id, int64_t token_count);
MADRONA_NO_INLINE void scheduleInferencePrefill(
    Engine &ctx, InferenceRuntime &runtime, const InferenceConfigData &config);
MADRONA_NO_INLINE void scheduleInferenceDecode(
    Engine &ctx, InferenceRuntime &runtime, const InferenceConfigData &config);

}
