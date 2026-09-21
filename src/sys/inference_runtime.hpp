#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE void inferencePreUpdate(Engine &ctx);
MADRONA_NO_INLINE void inferencePostUpdate(Engine &ctx);
// A9 per-request stats split: markInferenceStatsRefresh bumps the refresh
// sequence at the legacy update points (end of pre / end of post); the
// per-request task-graph nodes call refreshInferenceStatsRow once per
// request slot right after each of those nodes.
MADRONA_NO_INLINE void markInferenceStatsRefresh(Engine &ctx);
MADRONA_NO_INLINE void refreshInferenceStatsRow(
    Engine &ctx, InferenceRequestSlot &slot);
MADRONA_NO_INLINE bool inferenceIsFinished(
    const InferenceRuntime &runtime, const InferenceConfigData &config);

}
