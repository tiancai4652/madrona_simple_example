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
        return false;
    }

    // Lock-free bounded append. Multiple NPUs call addEvent concurrently from
    // the same ParallelForNode, so the old device SpinLock is replaced by an
    // atomic slot claim. A stale read of a just-claimed slot is benign: the
    // array is zero-initialized and readers skip 0, so the event is seen on
    // a later scan. Compaction happens in the serial pruneSystemEvents step.
    madrona::AtomicU32Ref count_ref(events.count);
    uint32_t slot = count_ref.fetch_add_relaxed(1);
    if (slot >= MAX_SYSTEM_EVENTS) {
        count_ref.fetch_sub<madrona::sync::relaxed>(1);
        madrona::AtomicU32Ref(events.overflow_count).fetch_add_relaxed(1);
        return false;
    }

    events.times_ns[slot] = event_time_ns;
#ifdef __CUDA_ARCH__
    __threadfence();
#endif
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
             uint32_t flow_id,
             uint64_t comm_para)
{
    // Multiple ProcessComm_E entities belonging to the same NPU can execute
    // sys_checkFlow concurrently on GPU. Reserve a distinct inbox slot
    // atomically; the later serial createFlowsFromNpuRequests step sorts the
    // batch by flow key before consuming it, so allocation order is identical
    // on CPU and GPU.
    Entity npu_e = ctx.data().findNpuEntity(npu_id);
    if (npu_e == Entity::none()) {
        return;
    }

    NpuFlowInbox &inbox = ctx.get<NpuFlowInbox>(npu_e);
    madrona::AtomicU32Ref count_ref(inbox.count);
    uint32_t slot = count_ref.fetch_add_relaxed(1);
    if (slot >= MAX_FLOWS_PER_NPU) {
        count_ref.fetch_sub<madrona::sync::relaxed>(1);
        madrona::AtomicU32Ref(inbox.overflow_count).fetch_add_relaxed(1);
        return;
    }

    inbox.entries[slot] = NpuFlowInboxEntry {
        .comm_src = comm_src,
        .comm_dst = comm_dst,
        .flow_size = flow_size,
        .flow_id = flow_id,
        .comm_para = comm_para,
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
        flow.comm_para = record.comm_para;
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
