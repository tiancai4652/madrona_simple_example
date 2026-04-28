#include "sim.hpp"
#include "sim_debug.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {
void Sim::loadFlow(Engine &ctx)
{
    (void)ctx;

    if (network == nullptr) {
        FATAL("Network input is missing");
    }

    int32_t flow_count = network->numFlows;
    if (flow_count > MAX_FLOWS) {
        FATAL("Flow count exceeds MAX_FLOWS");
    }

    if (flow_count > 0) {
        FlowId min_flow_id = network->flows[0].id;
        FlowId max_flow_id = network->flows[0].id;
        for (int32_t i = 0; i < flow_count; i++) {
            if (network->flows[i].id < min_flow_id) {
                min_flow_id = network->flows[i].id;
            }
            if (network->flows[i].id > max_flow_id) {
                max_flow_id = network->flows[i].id;
            }
        }

        int64_t lookup_span =
            (int64_t)max_flow_id - (int64_t)min_flow_id + 1;
        if (lookup_span > 0 && lookup_span <= MAX_FLOWS) {
            flowLookupBase = min_flow_id;
            flowLookupSpan = (int32_t)lookup_span;
            for (int32_t i = 0; i < flowLookupSpan; i++) {
                flowDefSlotLookup[i] = -1;
                flowRouteSlotLookup[i] = -1;
            }
        } else {
            flowLookupBase = 0;
            flowLookupSpan = 0;
        }
    }

    numFlowDefs = flow_count;
    numPendingFlows = flow_count;

    for (int32_t i = 0; i < flow_count; i++) {
        const FlowDef &flow = network->flows[i];
        if (i > 0 && flow.start_time < network->flows[i - 1].start_time) {
            FATAL("Flow input must be sorted by nondecreasing start_time");
        }

        flowDefs[i] = flow;
        pendingFlows[i] = flow;
        if (flowLookupSpan > 0) {
            flowDefSlotLookup[flow.id - flowLookupBase] = i;
        }
    }

    if constexpr (init_log_compiled_in) {
        if (init_log_print_enabled) {
            printInitFlowLog(*this);
        }
    }
}

}
