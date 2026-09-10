#pragma once

#include <madrona/macros.hpp>

#include "serving_types.hpp"

namespace madsimple {

MADRONA_NO_INLINE int32_t selectServingWorker(
    ServingRuntime &runtime, const InferenceConfigData &config,
    ServingStage stage);
MADRONA_NO_INLINE bool enqueueServingRequest(
    ServingWorker &worker, int32_t slot, int64_t tokens);

}
