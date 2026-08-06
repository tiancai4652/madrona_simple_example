#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

// -------------------------------------------------------------------------
// Persistent per-flow send/recv pairing state (keyed by comm_para).
// Replaces the (src,dst)-keyed send_recv_map_recvend slot semantics so that
// concurrent flows between the same pair are distinguished and a RECV that
// fires after its SEND flow already completed can still match.
// -------------------------------------------------------------------------

int32_t Sim::findFlowPairStateSlot(uint64_t comm_para,
                                   uint64_t comm_src,
                                   uint64_t comm_dst) const
{
    if (comm_para == 0) {
        return -1;
    }
    uint64_t hash = comm_para;
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    hash ^= comm_src * 0x9e3779b97f4a7c15ULL;
    hash ^= comm_dst * 0xbf58476d1ce4e5b9ULL;
    int32_t start = (int32_t)(hash & (MAX_FLOW_PAIR_STATES - 1));
    for (int32_t probe = 0; probe < MAX_FLOW_PAIR_STATES; probe++) {
        int32_t slot = (start + probe) & (MAX_FLOW_PAIR_STATES - 1);
        if (flow_pair_states[slot].comm_para == 0) {
            return slot;   // empty slot (caller decides to insert)
        }
        if (flow_pair_states[slot].comm_para == comm_para &&
            flow_pair_states[slot].comm_src == comm_src &&
            flow_pair_states[slot].comm_dst == comm_dst) {
            return slot;
        }
    }
    return -1;   // table full
}

// RECV side: return 1 if the flow's SEND has already completed (state 2) and
// reset the slot; otherwise register this RECV as waiting (state 1) and
// return 0. Keyed by the (comm_para, src, dst) triple so that workloads
// without globally-unique comm_para (e.g. multiverse comm_para=3 constant)
// still get one slot per (src,dst), while Huawei's unique flow_ids are
// distinguished per flow.
int32_t Sim::claimRecvDone(uint64_t comm_para, uint64_t comm_src, uint64_t comm_dst)
{
    flow_pair_lock.lock();
    int32_t slot = findFlowPairStateSlot(comm_para, comm_src, comm_dst);
    int32_t result = 0;
    if (slot >= 0) {
        if (flow_pair_states[slot].comm_para != comm_para) {
            flow_pair_states[slot] = FlowPairStateSlot {
                .comm_para = comm_para,
                .comm_src = comm_src,
                .comm_dst = comm_dst,
                .state = 1,
            };
        } else if (flow_pair_states[slot].state == 2) {
            // Consumed: keep the key so the linear-probe chain is not
            // broken by an empty hole shadowing later slots of this key.
            flow_pair_states[slot].state = 0;
            result = 1;
        } else if (flow_pair_states[slot].state == 0) {
            flow_pair_states[slot].state = 1;
        }
    }
    flow_pair_lock.unlock();
    return result;
}

// SEND side: mark the flow as completed (state 2). If the RECV was already
// waiting (state 1), it will be consumed by the RECV poll in the next system
// phase.
void Sim::markSendDone(uint64_t comm_para, uint64_t comm_src, uint64_t comm_dst)
{
    if (comm_para == 0) {
        return;
    }
    flow_pair_lock.lock();
    int32_t slot = findFlowPairStateSlot(comm_para, comm_src, comm_dst);
    if (slot >= 0) {
        if (flow_pair_states[slot].comm_para != comm_para) {
            flow_pair_states[slot] = FlowPairStateSlot {
                .comm_para = comm_para,
                .comm_src = comm_src,
                .comm_dst = comm_dst,
                .state = 2,
            };
        } else {
            flow_pair_states[slot].comm_src = comm_src;
            flow_pair_states[slot].comm_dst = comm_dst;
            flow_pair_states[slot].state = 2;
        }
    }
    flow_pair_lock.unlock();
}

namespace {

inline int32_t tagLookupMask()
{
    return MAX_PORT_TAG_LOOKUP - 1;
}

inline int32_t probeDistance(int32_t home_slot, int32_t slot)
{
    return (slot - home_slot + MAX_PORT_TAG_LOOKUP) & tagLookupMask();
}

inline int32_t hashTagLookupKey(FlowId flow_id)
{
    uint64_t key = (uint64_t)flow_id;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (int32_t)(key & (uint64_t)tagLookupMask());
}

inline int32_t flowMetaLookupMask()
{
    return MAX_FLOW_META_LOOKUP - 1;
}

inline int32_t hashFlowMetaLookupKey(FlowId flow_id)
{
    uint64_t key = (uint64_t)flow_id;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (int32_t)(key & (uint64_t)flowMetaLookupMask());
}

inline int32_t flowMetaProbeDistance(int32_t home_slot, int32_t slot)
{
    return (slot - home_slot + MAX_FLOW_META_LOOKUP) &
        flowMetaLookupMask();
}

inline bool tagLookupMatches(const PortTagLookupEntry &entry, FlowId flow_id)
{
    return entry.flow_id == flow_id;
}

MADRONA_NO_INLINE void retireFlowMeta(Context &ctx,
                                      Sim &sim,
                                      FlowId flow_id,
                                      Entity flow_entity)
{
    if (flow_entity == Entity::none()) {
        return;
    }

    FlowRuntimeState &runtime = ctx.get<FlowRuntimeState>(flow_entity);
    FlowCounters &counters = ctx.singleton<FlowCounters>();
    if (runtime.completed != 0) {
        return;
    }
    if (runtime.route_active != 0 && counters.numFlowRoutes > 0) {
        counters.numFlowRoutes -= 1;
    }
    if (counters.numFlowDefs > 0) {
        counters.numFlowDefs -= 1;
    }
    runtime.pending = 0;
    runtime.active = 0;
    runtime.completed = 1;
    runtime.route_active = 0;
    sim.removeFlowMetaLookup(flow_id);
}

}

MADRONA_NO_INLINE int32_t Sim::findFlowMetaLookupSlot(
    FlowId flow_id) const
{
    int32_t slot = hashFlowMetaLookupKey(flow_id);
    while (flowMetaLookupIds[slot] != -1) {
        if (flowMetaLookupIds[slot] == flow_id) {
            return slot;
        }
        slot = (slot + 1) & flowMetaLookupMask();
    }

    return -1;
}

MADRONA_NO_INLINE void Sim::insertFlowMetaLookup(FlowId flow_id,
                                                 Entity entity)
{
    int32_t slot = hashFlowMetaLookupKey(flow_id);
    while (flowMetaLookupIds[slot] != -1 &&
           flowMetaLookupIds[slot] != flow_id) {
        slot = (slot + 1) & flowMetaLookupMask();
    }

    flowMetaLookupIds[slot] = flow_id;
    flowMetaEntityLookup[slot] = entity;
}

MADRONA_NO_INLINE void Sim::removeFlowMetaLookup(FlowId flow_id)
{
    int32_t slot = findFlowMetaLookupSlot(flow_id);
    if (slot < 0) {
        return;
    }

    int32_t hole = slot;
    int32_t next = (hole + 1) & flowMetaLookupMask();

    while (flowMetaLookupIds[next] != -1) {
        int32_t home = hashFlowMetaLookupKey(flowMetaLookupIds[next]);
        if (flowMetaProbeDistance(home, next) >
            flowMetaProbeDistance(home, hole)) {
            flowMetaLookupIds[hole] = flowMetaLookupIds[next];
            flowMetaEntityLookup[hole] = flowMetaEntityLookup[next];
            hole = next;
        }
        next = (next + 1) & flowMetaLookupMask();
    }

    flowMetaLookupIds[hole] = -1;
    flowMetaEntityLookup[hole] = Entity::none();
}

MADRONA_NO_INLINE Entity Sim::findFlowMetaEntity(Context &ctx,
                                                 FlowId flow_id) const
{
    (void)ctx;
    int32_t lookup_idx = findFlowMetaLookupSlot(flow_id);
    if (lookup_idx >= 0) {
        return flowMetaEntityLookup[lookup_idx];
    }

    return Entity::none();
}

MADRONA_NO_INLINE const FlowDef *Sim::getFlowDef(Context &ctx,
                                                 FlowId flow_id) const
{
    Entity flow_entity = findFlowMetaEntity(ctx, flow_id);
    if (flow_entity == Entity::none()) {
        return nullptr;
    }

    const FlowRuntimeState &runtime =
        ctx.get<FlowRuntimeState>(flow_entity);
    if (runtime.completed != 0) {
        return nullptr;
    }

    return &ctx.get<FlowDef>(flow_entity);
}

MADRONA_NO_INLINE int32_t Sim::findTagLookupSlot(
    const PortTagLookup &lookup,
    FlowId flow_id) const
{
    int32_t slot = hashTagLookupKey(flow_id);
    int32_t start_slot = slot;

    while (lookup.entries[slot].flow_id != -1) {
        if (tagLookupMatches(lookup.entries[slot], flow_id)) {
            return slot;
        }

        slot = (slot + 1) & tagLookupMask();
        if (slot == start_slot) {
            break;
        }
    }

    return -1;
}

MADRONA_NO_INLINE void Sim::insertTagLookup(
    PortTagLookup &lookup,
    FlowId flow_id,
    Entity entity)
{
    int32_t slot = hashTagLookupKey(flow_id);

    while (lookup.entries[slot].flow_id != -1 &&
           !tagLookupMatches(lookup.entries[slot], flow_id)) {
        slot = (slot + 1) & tagLookupMask();
    }

    lookup.entries[slot].flow_id = flow_id;
    lookup.entries[slot].entity = entity;
}

MADRONA_NO_INLINE void Sim::removeTagLookup(PortTagLookup &lookup,
                                            FlowId flow_id)
{
    int32_t slot = findTagLookupSlot(lookup, flow_id);
    if (slot < 0) {
        return;
    }

    int32_t hole = slot;
    int32_t next = (hole + 1) & tagLookupMask();

    while (lookup.entries[next].flow_id != -1) {
        int32_t home = hashTagLookupKey(lookup.entries[next].flow_id);
        if (probeDistance(home, next) > probeDistance(home, hole)) {
            lookup.entries[hole] = lookup.entries[next];
            hole = next;
        }
        next = (next + 1) & tagLookupMask();
    }

    lookup.entries[hole] = PortTagLookupEntry {};
}

MADRONA_NO_INLINE Entity Sim::findTag(
    Context &ctx, int32_t port_id, FlowId flow_id) const
{
    if (port_id < 0 || port_id >= numPorts) {
        return Entity::none();
    }

    Entity port_e = portEntities[port_id];
    if (port_e == Entity::none()) {
        return Entity::none();
    }

    const PortTagLookup &lookup = ctx.get<PortTagLookup>(port_e);
    int32_t slot = findTagLookupSlot(lookup, flow_id);
    if (slot >= 0) {
        Entity tag_e = lookup.entries[slot].entity;
        if (tag_e != Entity::none()) {
            const FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            if (tag.port_id == port_id && tag.flow_id == flow_id) {
                return tag_e;
            }
        }
    }

    const PortTagList &ptl = ctx.get<PortTagList>(port_e);
    for (int32_t i = 0; i < ptl.count; i++) {
        Entity tag_e = ptl.tags[i];
        if (tag_e == Entity::none()) {
            continue;
        }

        const FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
        if (tag.flow_id == flow_id) {
            return tag_e;
        }
    }

    return Entity::none();
}

MADRONA_NO_INLINE void Sim::recordFlowCompletion(
    Context &ctx, FlowId flow_id, Time end_time)
{
    Entity flow_entity = findFlowMetaEntity(ctx, flow_id);
    if (flow_entity == Entity::none()) {
        return;
    }

    const FlowDef &flow = ctx.get<FlowDef>(flow_entity);
    FlowRuntimeState &runtime = ctx.get<FlowRuntimeState>(flow_entity);
    FlowCounters &counters = ctx.singleton<FlowCounters>();

    if (runtime.completed != 0) {
        if (end_time > runtime.completion_record.end_time) {
            runtime.completion_record.end_time = end_time;
        }
        return;
    }

    if (counters.numFlowCompletions < MAX_FLOW_COMPLETIONS) {
        counters.numFlowCompletions += 1;
    } else {
        retireFlowMeta(ctx, *this, flow_id, flow_entity);
        return;
    }

    runtime.completion_record = FlowCompletionRecord {
        .flow_id = flow_id,
        .src_node = flow.src_node,
        .dst_node = flow.dst_node,
        .size = flow.size,
        .start_time = flow.start_time,
        .end_time = end_time,
        .priority = flow.priority,
    };

    // NPU-owned dynamic flow: mirror the completion into that NPU's own
    // bounded finished-flow mailbox, remove it from the NPU's active list,
    // and hand the FlowMeta entity straight back to the NPU's own pool for
    // reuse. This runs from the (already serial) network-side completion
    // flush, so touching another entity's (the owning NPU's) components
    // here is safe -- it is the one deliberate serialization point in the
    // per-NPU design, and it is O(that NPU's MAX_FLOWS_PER_NPU) at worst,
    // not O(total flows ever created).
    if (runtime.owner_npu_id >= 0) {
        // The FlowMeta entity itself goes back into the NPU's pool below, so
        // keep a durable copy of the FCT record for the Python-visible
        // flow_completion(idx) export.
        DynamicFlowCompletionLog &dyn_log =
            ctx.singleton<DynamicFlowCompletionLog>();
        if (dyn_log.count < MAX_DYNAMIC_FLOW_COMPLETIONS) {
            dyn_log.records[dyn_log.count++] = runtime.completion_record;
        } else {
            dyn_log.overflow_count += 1;
        }

        Entity npu_entity = findNpuEntity((uint32_t)runtime.owner_npu_id);
        if (npu_entity != Entity::none()) {
            NpuFlowFinishedList &finished =
                ctx.get<NpuFlowFinishedList>(npu_entity);
            if (finished.count < MAX_FLOWS_PER_NPU) {
                finished.flows[finished.count++] = SysFlowRecord {
                    .owner_npu_id = (uint32_t)runtime.owner_npu_id,
                    .flow_id = (uint32_t)flow_id,
                    .comm_src = (uint64_t)flow.src_node,
                    .comm_dst = (uint64_t)flow.dst_node,
                    .flow_size = (uint64_t)flow.size,
                    .comm_para = flow.comm_para,
                    .start_time_ns =
                        (uint64_t)(flow.start_time * 1000000.0 + 0.5),
                    .end_time_ns = (uint64_t)(end_time * 1000000.0 + 0.5),
                };
            } else {
                finished.overflow_count += 1;
            }

            markSendDone(flow.comm_para, (uint64_t)flow.src_node,
                         (uint64_t)flow.dst_node);

            NpuFlowActiveList &active =
                ctx.get<NpuFlowActiveList>(npu_entity);
            for (uint32_t i = 0; i < active.count; i++) {
                if (active.active[i].flow_entity != flow_entity) {
                    continue;
                }
                active.active[i] = active.active[active.count - 1];
                active.count -= 1;
                break;
            }

            NpuFlowPool &pool = ctx.get<NpuFlowPool>(npu_entity);
            if (pool.free_count < MAX_FLOWS_PER_NPU) {
                pool.free_flows[pool.free_count++] = flow_entity;
            }
        }
    }

    retireFlowMeta(ctx, *this, flow_id, flow_entity);
}

MADRONA_NO_INLINE Entity Sim::findNpuEntity(uint32_t npu_id) const
{
    if (npu_id >= MAX_NPUS) {
        return Entity::none();
    }

    return npuEntities[npu_id];
}

MADRONA_NO_INLINE void Sim::createFlowsFromNpuRequests(Context &ctx)
{
    FlowCounters &counters = ctx.singleton<FlowCounters>();

    for (int32_t npu_id = 0; npu_id < numNpus; npu_id++) {
        Entity npu_entity = npuEntities[npu_id];
        if (npu_entity == Entity::none()) {
            continue;
        }

        NpuFlowInbox &inbox = ctx.get<NpuFlowInbox>(npu_entity);
        if (inbox.count == 0) {
            continue;
        }

        NpuFlowPool &pool = ctx.get<NpuFlowPool>(npu_entity);
        NpuFlowActiveList &active = ctx.get<NpuFlowActiveList>(npu_entity);

        for (uint32_t i = 0; i < inbox.count; i++) {
            const NpuFlowInboxEntry &req = inbox.entries[i];

            if (pool.free_count == 0) {
                inbox.overflow_count += 1;
                continue;
            }
            if (active.count >= MAX_FLOWS_PER_NPU) {
                inbox.overflow_count += 1;
                continue;
            }

            Entity flow_entity = pool.free_flows[--pool.free_count];
            pool.free_flows[pool.free_count] = Entity::none();

            FlowDef flow {
                .id = (FlowId)req.flow_id,
                .src_node = (NodeId)req.comm_src,
                .dst_node = (NodeId)req.comm_dst,
                .size = (Bytes)req.flow_size,
                .start_time = now,
                .priority = 0,
                .comm_para = req.comm_para,
            };
            ctx.get<FlowDef>(flow_entity) = flow;
            ctx.get<FlowRouteState>(flow_entity) = FlowRouteState {};
            ctx.get<FlowRuntimeState>(flow_entity) = FlowRuntimeState {
                .pending = 1,
                .active = 0,
                .completed = 0,
                .route_active = 0,
                .owner_npu_id = npu_id,
                .completion_record = FlowCompletionRecord {},
            };
            // flow_order = -1: this entity is NPU-owned, not part of the
            // static flowMetaEntities[]/pendingFlowCursor window (see
            // FlowScheduleState comment in sim.hpp).
            ctx.get<FlowScheduleState>(flow_entity) = FlowScheduleState {
                .flow_order = -1,
                .ready_now = 0,
                .prepared = 0,
                .port_path_len = 0,
                .port_path = {},
                .prepared_event = DelayedEvent {},
            };

            // The flow_id -> FlowMeta lookup table stays a single shared
            // structure (network-side routing needs to resolve any flow_id
            // to its entity regardless of owner); insertion here is safe
            // because createFlowsFromNpuRequests itself is a serial singleton
            // step, so this loop never races with another NPU's insert.
            insertFlowMetaLookup(flow.id, flow_entity);

            active.active[active.count++] = NpuActiveFlow {
                .flow_entity = flow_entity,
                .flow_id = req.flow_id,
            };

            // NOTE: counters.numPendingFlows is intentionally left untouched
            // here. It is exclusively the static flow-file scheduling
            // window's size (see schedulePendingFlows's
            // pendingFlowCursor + numPendingFlows window math in
            // sim_systems_schedule.cpp); NPU-owned flows are scheduled
            // through the independent scheduleNpuFlows() path below and
            // must never grow that window, or schedulePendingFlows would
            // start reading past the static flowMetaEntities[] contents it
            // actually owns.
            counters.numFlowDefs += 1;
        }

        inbox.count = 0;
    }
}

MADRONA_NO_INLINE void Sim::pruneSystemEvents(Context &ctx)
{
    SystemEventQueue &events = ctx.singleton<SystemEventQueue>();
    uint64_t now_ns = (uint64_t)(now * 1000000.0 + 0.5);
    uint32_t out = 0;
    for (uint32_t i = 0; i < events.count; i++) {
        uint64_t event_ns = events.times_ns[i];
        if (event_ns > now_ns) {
            events.times_ns[out++] = event_ns;
        }
    }
    events.count = out;
}

MADRONA_NO_INLINE void Sim::refreshTagCounters(Context &ctx)
{
    SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();
    int32_t active_tags = 0;
    int32_t source_tags = 0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        active_tags += ctx.get<PortTagList>(port_e).count;
        source_tags += ctx.get<PortSourceTagList>(port_e).count;
    }

    runtime.numActiveTags = active_tags;
    runtime.numSourceTags = source_tags;
}

}
