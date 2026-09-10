#include "op_resolver.hpp"

namespace madsimple {

namespace {

inline uint64_t ceilDiv(uint64_t numerator, uint64_t denominator)
{
    return numerator / denominator + (numerator % denominator != 0);
}

inline uint64_t saturatingMul(uint64_t a, uint64_t b)
{
    if (a != 0 && b > UINT64_MAX / a) {
        return UINT64_MAX;
    }
    return a * b;
}

inline uint64_t saturatingScale(uint64_t value, uint64_t tokens,
                                uint64_t reference)
{
    return ceilDiv(saturatingMul(value, tokens), reference);
}

}

void resolveServingOp(const InferenceConfigData &config,
                      const WorkloadParamsTableData &profiles,
                      const ServingNpuExecution &execution,
                      ChakraNode &node)
{
    if (config.data[IC_ENABLED] == 0 || execution.active == 0) {
        return;
    }

    int32_t count = static_cast<int32_t>(config.data[IC_WORKLOAD_PARAMS_TABLE_COUNT]);
    if (count < 0) {
        count = 0;
    } else if (count > MAX_WORKLOAD_PARAMS_TABLE_ENTRIES) {
        count = MAX_WORKLOAD_PARAMS_TABLE_ENTRIES;
    }
    int32_t best = -1;
    int32_t best_specificity = -1;
    for (int32_t i = 0; i < count; i++) {
        const int64_t *profile = profiles.data[i];
        if (profile[WPT_STAGE] != execution.stage ||
            profile[WPT_NODE_TYPE] != static_cast<int64_t>(node.type)) {
            continue;
        }
        const bool batch_exact =
            profile[WPT_BATCH_SIZE] == execution.request_count;
        const bool sequence_exact =
            profile[WPT_SEQUENCE_TOKENS] == execution.token_count;
        if ((!batch_exact && profile[WPT_BATCH_SIZE] != -1) ||
            (!sequence_exact && profile[WPT_SEQUENCE_TOKENS] != -1)) {
            continue;
        }
        const int32_t specificity =
            static_cast<int32_t>(batch_exact) +
            static_cast<int32_t>(sequence_exact);
        if (specificity > best_specificity) {
            best = i;
            best_specificity = specificity;
        }
    }
    if (best >= 0) {
        const int64_t *profile = profiles.data[best];
        const uint64_t duration_ns =
            static_cast<uint64_t>(profile[WPT_DURATION_NS]);
        uint64_t duration_us = ceilDiv(duration_ns, 1000);
        if (node.type == ChakraNodeType::COMP_NODE && duration_us == 0) {
            duration_us = 1;
        }
        node.durationMicros = static_cast<uint32_t>(
            duration_us > UINT32_MAX ? UINT32_MAX : duration_us);
        node.comm_size =
            static_cast<uint64_t>(profile[WPT_COMM_SIZE_BYTES]);
        return;
    }

    int64_t reference = execution.stage ==
        static_cast<int32_t>(ServingStage::Prefill)
        ? config.data[IC_PREFILL_REFERENCE_TOKENS]
        : config.data[IC_DECODE_REFERENCE_TOKENS];
    if (reference <= 0) {
        reference = 1;
    }
    int64_t tokens = execution.token_count > 0 ? execution.token_count : 1;

    uint64_t duration = saturatingScale(
        static_cast<uint64_t>(node.durationMicros),
        static_cast<uint64_t>(tokens),
        static_cast<uint64_t>(reference));
    node.durationMicros = static_cast<uint32_t>(duration > UINT32_MAX
        ? UINT32_MAX : (duration == 0 ? 1 : duration));

    if (node.comm_size > 0) {
        uint64_t scaled = saturatingScale(
            node.comm_size, static_cast<uint64_t>(tokens),
            static_cast<uint64_t>(reference));
        node.comm_size = scaled == 0 ? 1 : scaled;
    }
}

}
