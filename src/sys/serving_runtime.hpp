#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE void servingPreUpdate(Engine &ctx);
MADRONA_NO_INLINE void servingPostUpdate(Engine &ctx);
MADRONA_NO_INLINE void updateServingStats(Engine &ctx);
MADRONA_NO_INLINE bool servingIsFinished(
    const ServingRuntime &runtime, const InferenceConfigData &config);

}
