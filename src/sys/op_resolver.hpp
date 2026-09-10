#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE void resolveServingOp(
    const InferenceConfigData &config,
    const WorkloadParamsTableData &profiles,
    const ServingNpuExecution &execution,
    ChakraNode &node);

}
