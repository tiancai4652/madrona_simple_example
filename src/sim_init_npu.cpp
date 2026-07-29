#include "sim.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

// Creates one FakeSystemArch entity for NPU `npu_id` and preallocates its
// NpuFlowPool. This mirrors initPortQueues()'s PortTagPool preallocation
// (sim_init_port.cpp): every FlowMeta entity this NPU will ever use is
// created exactly once here, up front, single-threaded. From then on,
// setFlow()/createFlowsFromNpuRequests()/recordFlowCompletion() only ever
// pop from / push back into this NPU's own free list -- no ctx.makeEntity
// or ctx.destroyEntity call happens on the per-step hot path.
//
// `src_npu`/`dst_npu` seed this NPU's FakeSystemDriver state machine (see
// fake_system.cpp) with the traffic pattern it will drive for the
// validation run; a real system-layer driver would instead be populated
// from Chakra task data and would leave the Npu* bookkeeping components
// exactly as initialized here.
MADRONA_NO_INLINE void Sim::createNpu(Context &ctx,
                                      uint32_t npu_id,
                                      uint64_t src_npu,
                                      uint64_t dst_npu)
{
    if (npu_id >= MAX_NPUS) {
        FATAL("npu_id exceeds MAX_NPUS");
    }

    Entity npu_entity = ctx.makeEntity<FakeSystemArch>();

    ctx.get<FakeSystemStats>(npu_entity) = FakeSystemStats {
        .enabled = 1,
        .state = FakeSystemState::Init,
        .npu_id = npu_id,
        .src_npu = src_npu,
        .dst_npu = dst_npu,
        .flow_id_base = 900000 + npu_id * 1000,
    };

    ctx.get<NpuFlowInbox>(npu_entity) = NpuFlowInbox {};

    NpuFlowPool pool {};
    pool.free_count = MAX_FLOWS_PER_NPU;
    for (uint32_t i = 0; i < MAX_FLOWS_PER_NPU; i++) {
        Entity flow_entity = ctx.makeEntity<FlowMeta>();
        // Idle default: pending == 0 so preparePendingFlowMeta/
        // scheduleNpuFlows ignores this entity until
        // createFlowsFromNpuRequests actually assigns it to a real request.
        ctx.get<FlowDef>(flow_entity) = FlowDef {};
        ctx.get<FlowRouteState>(flow_entity) = FlowRouteState {};
        ctx.get<FlowRuntimeState>(flow_entity) = FlowRuntimeState {
            .pending = 0,
            .active = 0,
            .completed = 0,
            .route_active = 0,
            .owner_npu_id = -1,
            .completion_record = FlowCompletionRecord {},
        };
        ctx.get<FlowScheduleState>(flow_entity) = FlowScheduleState {
            .flow_order = -1,
        };
        pool.free_flows[i] = flow_entity;
    }
    ctx.get<NpuFlowPool>(npu_entity) = pool;

    ctx.get<NpuFlowActiveList>(npu_entity) = NpuFlowActiveList {};
    ctx.get<NpuFlowFinishedList>(npu_entity) = NpuFlowFinishedList {};

    npuEntities[npu_id] = npu_entity;
    if ((int32_t)(npu_id + 1) > numNpus) {
        numNpus = (int32_t)(npu_id + 1);
    }
}

MADRONA_NO_INLINE Entity Sim::findNpuEntity(uint32_t npu_id) const
{
    if (npu_id >= MAX_NPUS) {
        return Entity::none();
    }
    return npuEntities[npu_id];
}

}
