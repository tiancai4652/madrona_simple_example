#pragma once

#include <madrona/macros.hpp>

#include "inference_types.hpp"

namespace madsimple {

MADRONA_NO_INLINE int32_t selectInferenceWorker(
    InferenceRuntime &runtime, const InferenceConfigData &config,
    InferenceStage stage);
MADRONA_NO_INLINE bool enqueueInferenceRequest(
    InferenceWorker &worker, int32_t slot, int64_t tokens);

}
