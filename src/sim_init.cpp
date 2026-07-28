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

    FlowCounters &flow_counters = ctx.singleton<FlowCounters>();
    flow_counters = FlowCounters {
        .numFlowDefs = flow_count,
        .numPendingFlows = flow_count,
        .pendingFlowCursor = 0,
        .numFlowRoutes = 0,
        .numFlowCompletions = 0,
    };

    ctx.singleton<SimRuntimeState>() = SimRuntimeState {
        .numDelayedEvents = 0,
        .numActiveTags = 0,
        .numSourceTags = 0,
        .pfcControlEventsSeen = 0,
        .cachedDrainPortID = -1,
        .cachedNextDelayedGap = timerInactiveSentinel(),
        .cachedNextBacklogGap = timerInactiveSentinel(),
        .cachedNextPfcPauseGap = timerInactiveSentinel(),
        .cachedNextPfcResumeGap = timerInactiveSentinel(),
        .cachedNextDrainTime = timerInactiveSentinel(),
        .cachedNextFinishTime = timerInactiveSentinel(),
        .nextDT = 0.0,
    };

    for (int32_t i = 0; i < flow_count; i++) {
        const FlowDef &flow = network->flows[i];
        if (i > 0 && flow.start_time < network->flows[i - 1].start_time) {
            FATAL("Flow input must be sorted by nondecreasing start_time");
        }

        Entity flow_entity = ctx.makeEntity<FlowMeta>();
        ctx.get<FlowDef>(flow_entity) = flow;
        ctx.get<FlowRouteState>(flow_entity) = FlowRouteState {};
        ctx.get<FlowRuntimeState>(flow_entity) = FlowRuntimeState {
            .pending = 1,
            .active = 0,
            .completed = 0,
            .route_active = 0,
            .owner_npu_id = -1,
            .completion_record = FlowCompletionRecord {},
        };
        ctx.get<FlowScheduleState>(flow_entity) = FlowScheduleState {
            .flow_order = i,
            .ready_now = 0,
            .prepared = 0,
            .port_path_len = 0,
            .port_path = {},
            .prepared_event = DelayedEvent {},
        };
        flowMetaEntities[i] = flow_entity;
        insertFlowMetaLookup(flow.id, flow_entity);
    }
    numFlowMetaEntities = flow_count;

    if constexpr (init_log_compiled_in) {
        if (init_log_print_enabled) {
            printInitFlowLog(*this);
        }
    }
}

}
