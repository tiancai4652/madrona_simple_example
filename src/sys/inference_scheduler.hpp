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

// A10 per-NPU copy pass: one task-graph instance per NPU. Scans the
// InferenceResetRoster (written by the scheduling decision inside
// inferencePreUpdate, cleared at its head every frame); when this NPU's id
// appears, copies the template answer sheet and resets this NPU's own
// components -- exactly what the legacy inline resetNpuForReplay did at
// decision time. Guards mirror inferencePreUpdate (enabled && initialized
// && !failed) so failed/disabled frames skip the copy exactly like the
// legacy decision path did.
MADRONA_NO_INLINE void inferenceResetLocal(
    Engine &ctx, uint32_t npu_id, ChakraNodes &chakraNodes,
    ChakraTemplateNodes &templ, HardwareResource &hardwareResource,
    ProcessingCompTask &processingCompTask,
    ProcessingCommTasks &processingCommTasks,
    OneNPUFinishedFlag &oneNPUFinishedFlag, ChakraNodesForNoDP &noDP,
    ChakraReadyState &readyState, InferenceNpuExecution &execution);

}
