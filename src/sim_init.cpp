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

    numFlowDefs = flow_count;
    numPendingFlows = flow_count;

    for (int32_t i = 0; i < flow_count; i++) {
        const FlowDef &flow = network->flows[i];
        if (i > 0 && flow.start_time < network->flows[i - 1].start_time) {
            FATAL("Flow input must be sorted by nondecreasing start_time");
        }

        flowDefs[i] = flow;
        pendingFlows[i] = flow;
    }

    if constexpr (init_log_compiled_in) {
        if (init_log_print_enabled) {
            printInitFlowLog(*this);
        }
    }
}

}
