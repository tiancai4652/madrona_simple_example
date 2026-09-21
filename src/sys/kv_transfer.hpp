#pragma once

#include "../sim.hpp"

namespace madsimple {

MADRONA_NO_INLINE bool beginInferenceKvTransfer(
    Engine &ctx, InferenceRuntime &runtime,
    const InferenceConfigData &config, int32_t slot);
// GQA same-head pairing layout check: true when p_width/d_width each form
// an integer multiple relationship with kv_heads and with each other.
bool gqaInferenceLayoutValid(int64_t p_width, int64_t d_width,
                            int64_t kv_heads);

// A1 per-NPU split of the legacy collectInferenceKvCompletions singleton.
// Parallel draft pass: one instance per NPU scans that NPU's own
// NpuFlowFinishedList and records raw KV-completion tickets (slot, shard,
// end_time) into the NPU's own NpuKvCollectDraft -- zero writes to the
// shared InferenceRuntime. Entry filters that depend only on the list
// entry (KV flow-id marker, slot/shard bounds) run here; every check that
// touches request state stays in the commit pass so the replay reproduces
// the legacy state machine exactly.
MADRONA_NO_INLINE void collectInferenceKvDrafts(
    Engine &ctx, uint32_t npu_id, const NpuFlowFinishedList &finished,
    NpuKvCollectDraft &draft);
// Serial commit pass: replays every NPU's tickets in npu_id ascending
// order -- byte-identical to the legacy singleton's npu-major iteration
// order -- applying the same state checks and mutations (kv_done_mask OR,
// t_kv_done max, kv_done count, inflight_kv, last-shard handoff including
// the D-worker queue enqueue and inferenceHandoffPending). No-ops when
// there are no drafts. Clears the drafts.
MADRONA_NO_INLINE void commitInferenceKvCollect(Engine &ctx);

}
