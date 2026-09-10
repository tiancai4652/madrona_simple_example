#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE void inferencePreUpdate(Engine &ctx);
MADRONA_NO_INLINE void inferencePostUpdate(Engine &ctx);
MADRONA_NO_INLINE void updateInferenceStats(Engine &ctx);
MADRONA_NO_INLINE bool inferenceIsFinished(
    const InferenceRuntime &runtime, const InferenceConfigData &config);

}
