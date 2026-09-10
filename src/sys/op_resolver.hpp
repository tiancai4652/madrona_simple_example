#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE void resolveInferenceOp(
    const InferenceConfigData &config,
    const WorkloadParamsTableData &profiles,
    const InferenceNpuExecution &execution,
    ChakraNode &node);

}
