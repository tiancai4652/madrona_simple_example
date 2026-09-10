#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE bool beginServingKvTransfer(
    Engine &ctx, ServingRuntime &runtime,
    const InferenceConfigData &config, int32_t slot);
// GQA same-head pairing layout check: true when p_width/d_width each form
// an integer multiple relationship with kv_heads and with each other.
bool gqaServingLayoutValid(int64_t p_width, int64_t d_width,
                           int64_t kv_heads);
MADRONA_NO_INLINE void collectServingKvCompletions(Engine &ctx);

}
