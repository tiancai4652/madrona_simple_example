#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

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

inline bool tagLookupMatches(const PortTagLookupEntry &entry, FlowId flow_id)
{
    return entry.flow_id == flow_id;
}

MADRONA_NO_INLINE void retireFlowMeta(Context &ctx,
                                      Sim &sim,
                                      FlowId,
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
}

}

MADRONA_NO_INLINE int32_t Sim::flowLookupIndex(FlowId flow_id) const
{
    if (flowLookupSpan <= 0) {
        return -1;
    }

    int64_t lookup_idx = (int64_t)flow_id - (int64_t)flowLookupBase;
    if (lookup_idx < 0 || lookup_idx >= flowLookupSpan) {
        return -1;
    }

    return (int32_t)lookup_idx;
}

MADRONA_NO_INLINE Entity Sim::findFlowMetaEntity(Context &ctx,
                                                 FlowId flow_id) const
{
    int32_t lookup_idx = flowLookupIndex(flow_id);
    if (lookup_idx >= 0) {
        return flowMetaEntityLookup[lookup_idx];
    }

    if (network == nullptr) {
        return Entity::none();
    }

    for (int32_t i = 0; i < network->numFlows; i++) {
        Entity flow_entity = flowMetaEntities[i];
        if (flow_entity == Entity::none()) {
            continue;
        }

        const FlowDef &flow = ctx.get<FlowDef>(flow_entity);
        if (flow.id == flow_id) {
            return flow_entity;
        }
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

    retireFlowMeta(ctx, *this, flow_id, flow_entity);
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
