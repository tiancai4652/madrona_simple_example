#include "net_sys_interface.hpp"

namespace madsimple::net_sys_interface {

using madrona::Entity;

uint64_t msToNs(Time time_ms)
{
    if (time_ms <= 0.0) {
        return 0;
    }
    return (uint64_t)(time_ms * 1000000.0 + 0.5);
}

Time nsToMs(uint64_t time_ns)
{
    return (Time)time_ns / 1000000.0;
}

uint64_t getCurrentTime(Engine &ctx)
{
    return msToNs(ctx.data().now);
}

bool addEvent(Engine &ctx, uint64_t event_time_ns)
{
    uint64_t now_ns = getCurrentTime(ctx);
    SystemEventQueue &events = ctx.singleton<SystemEventQueue>();

    if (event_time_ns <= now_ns) {
        events.overflow_count += 1;
        return false;
    }

    events.lock.lock();
    if (events.count >= MAX_SYSTEM_EVENTS) {
        events.overflow_count += 1;
        events.lock.unlock();
        return false;
    }

    events.times_ns[events.count++] = event_time_ns;
    events.lock.unlock();
    return true;
}

bool isExistedFlow(Engine &ctx)
{
    const FlowCounters &counters = ctx.singleton<FlowCounters>();
    const SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();

    if (counters.numPendingFlows > 0 ||
        runtime.numActiveTags > 0 ||
        runtime.numDelayedEvents > 0) {
        return true;
    }

    // Bounded scan over NPUs (not over flows): each NPU's own inbox/pool/
    // active-list sizes are capped at MAX_FLOWS_PER_NPU, so this is cheap
    // regardless of how many flows have been materialized and retired over
    // the life of the run.
    const Sim &sim = ctx.data();
    for (int32_t npu_id = 0; npu_id < sim.numNpus; npu_id++) {
        Entity npu_e = sim.npuEntities[npu_id];
        if (npu_e == Entity::none()) {
            continue;
        }
        if (ctx.get<NpuFlowInbox>(npu_e).count > 0 ||
            ctx.get<NpuFlowActiveList>(npu_e).count > 0) {
            return true;
        }
    }

    return false;
}

void addSimtime(Engine &ctx, uint64_t delta_ns)
{
    if (delta_ns == 0) {
        return;
    }

    uint64_t now_ns = getCurrentTime(ctx);
    uint64_t actual_delta = delta_ns;
    const SystemEventQueue &events = ctx.singleton<SystemEventQueue>();
    for (uint32_t i = 0; i < events.count; i++) {
        uint64_t event_ns = events.times_ns[i];
        if (event_ns > now_ns) {
            uint64_t gap = event_ns - now_ns;
            if (gap < actual_delta) {
                actual_delta = gap;
            }
        }
    }

    ctx.data().now += nsToMs(actual_delta);
}

void setFlow(Engine &ctx,
             uint32_t npu_id,
             uint64_t comm_src,
             uint64_t comm_dst,
             uint64_t flow_size,
             uint32_t flow_id)
{
    // Design invariant: this must only ever be called by NPU `npu_id`'s
    // own step-system invocation (a per-NPU worker submitting a flow it
    // itself owns). Under that invariant, findNpuEntity(npu_id) resolves
    // back to the exact entity this call is already executing "as", so
    // writing into its own NpuFlowInbox touches only this entity's own
    // column storage -- no lock needed even when many NPUs call setFlow
    // concurrently from the same fakeSystemStepSystem ParallelForNode.
    Entity npu_e = ctx.data().findNpuEntity(npu_id);
    if (npu_e == Entity::none()) {
        return;
    }

    NpuFlowInbox &inbox = ctx.get<NpuFlowInbox>(npu_e);
    if (inbox.count >= MAX_FLOWS_PER_NPU) {
        inbox.overflow_count += 1;
        return;
    }

    inbox.entries[inbox.count++] = NpuFlowInboxEntry {
        .comm_src = comm_src,
        .comm_dst = comm_dst,
        .flow_size = flow_size,
        .flow_id = flow_id,
    };
}

uint32_t checkFlowFinish(Engine &ctx,
                         uint32_t npu_id,
                         SysFlow flows_finish[])
{
    return checkFlowFinish(ctx, npu_id, flows_finish,
                           MAX_FLOWS_PER_NPU);
}

uint32_t checkFlowFinish(Engine &ctx,
                         uint32_t npu_id,
                         SysFlow flows_finish[],
                         uint32_t max_flows_finish)
{
    Entity npu_e = ctx.data().findNpuEntity(npu_id);
    if (npu_e == Entity::none()) {
        return 0;
    }

    // Read-only scan over this NPU's own bounded finished-flow mailbox
    // (every entry already belongs to npu_id -- see recordFlowCompletion --
    // so unlike the old global-snapshot design there is no owner filtering
    // to do, and no other NPU's completions can ever appear here).
    const NpuFlowFinishedList &finished =
        ctx.get<NpuFlowFinishedList>(npu_e);
    uint32_t out_count = 0;
    for (uint32_t i = 0; i < finished.count && out_count < max_flows_finish;
         i++) {
        const SysFlowRecord &record = finished.flows[i];
        SysFlow &flow = flows_finish[out_count++];
        flow.id = record.flow_id;
        flow.comm_size = record.flow_size;
        flow.comm_src = record.comm_src;
        flow.comm_dst = record.comm_dst;
        flow.durationMicros = static_cast<uint32_t>(
            (record.end_time_ns - record.start_time_ns) / 1000);
        flow.state = TaskState::FINISH;
        flow.is_send = true;
    }

    return out_count;
}

void clearFlowFinishQueue(Engine &ctx, uint32_t npu_id)
{
    Entity npu_e = ctx.data().findNpuEntity(npu_id);
    if (npu_e == Entity::none()) {
        return;
    }

    ctx.get<NpuFlowFinishedList>(npu_e).count = 0;
}

}
