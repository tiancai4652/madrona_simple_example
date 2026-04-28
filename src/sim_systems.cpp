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
    return TAG_LOOKUP_CAPACITY - 1;
}

inline int32_t probeDistance(int32_t home_slot, int32_t slot)
{
    return (slot - home_slot + TAG_LOOKUP_CAPACITY) & tagLookupMask();
}

inline int32_t hashTagLookupKey(int32_t port_id, FlowId flow_id)
{
    uint64_t key =
        (((uint64_t)(uint32_t)port_id) << 32) ^ (uint64_t)flow_id;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (int32_t)(key & (uint64_t)tagLookupMask());
}

inline bool tagLookupMatches(const TagLookupEntry &entry,
                             int32_t port_id,
                             FlowId flow_id)
{
    return entry.port_id == port_id && entry.flow_id == flow_id;
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

MADRONA_NO_INLINE int32_t Sim::findSourceTagIndex(FlowId flow_id) const
{
    int32_t lookup_idx = flowLookupIndex(flow_id);
    if (lookup_idx >= 0) {
        int32_t slot = sourceTagSlotLookup[lookup_idx];
        if (slot >= 0 && slot < numSourceTags &&
            sourceTags[slot].flow_id == flow_id) {
            return slot;
        }
    }

    for (int32_t i = 0; i < numSourceTags; i++) {
        if (sourceTags[i].flow_id == flow_id) {
            return i;
        }
    }
    return -1;
}

MADRONA_NO_INLINE int32_t Sim::findIngressTagIndex(
    int32_t ingress_port_id, FlowId flow_id) const
{
    for (int32_t i = 0; i < numIngressTags; i++) {
        if (ingressTags[i].ingress_port_id == ingress_port_id &&
            ingressTags[i].flow_id == flow_id) {
            return i;
        }
    }
    return -1;
}

MADRONA_NO_INLINE int32_t Sim::findFlowDefSlot(FlowId flow_id) const
{
    int32_t lookup_idx = flowLookupIndex(flow_id);
    if (lookup_idx >= 0) {
        int32_t slot = flowDefSlotLookup[lookup_idx];
        if (slot >= 0 && slot < numFlowDefs &&
            flowDefs[slot].id == flow_id) {
            return slot;
        }
    }

    for (int32_t i = 0; i < numFlowDefs; i++) {
        if (flowDefs[i].id == flow_id) {
            return i;
        }
    }

    return -1;
}

MADRONA_NO_INLINE int32_t Sim::findFlowRouteSlot(FlowId flow_id) const
{
    int32_t lookup_idx = flowLookupIndex(flow_id);
    if (lookup_idx >= 0) {
        int32_t slot = flowRouteSlotLookup[lookup_idx];
        if (slot >= 0 && slot < numFlowRoutes &&
            flowRoutes[slot].flow_id == flow_id) {
            return slot;
        }
    }

    for (int32_t i = 0; i < numFlowRoutes; i++) {
        if (flowRoutes[i].flow_id == flow_id) {
            return i;
        }
    }

    return -1;
}

MADRONA_NO_INLINE int32_t Sim::findFlowCompletionSlot(FlowId flow_id) const
{
    int32_t lookup_idx = flowLookupIndex(flow_id);
    if (lookup_idx >= 0) {
        int32_t slot = flowCompletionSlotLookup[lookup_idx];
        if (slot >= 0 && slot < numFlowCompletions &&
            flowCompletions[slot].flow_id == flow_id) {
            return slot;
        }
    }

    for (int32_t i = 0; i < numFlowCompletions; i++) {
        if (flowCompletions[i].flow_id == flow_id) {
            return i;
        }
    }

    return -1;
}

MADRONA_NO_INLINE const FlowDef *Sim::getFlowDef(FlowId flow_id) const
{
    int32_t slot = findFlowDefSlot(flow_id);
    if (slot < 0 || slot >= numFlowDefs) {
        return nullptr;
    }

    return &flowDefs[slot];
}

MADRONA_NO_INLINE int32_t Sim::findTagLookupSlot(
    int32_t port_id, FlowId flow_id) const
{
    int32_t slot = hashTagLookupKey(port_id, flow_id);
    int32_t start_slot = slot;

    while (tagLookup[slot].port_id != -1) {
        if (tagLookupMatches(tagLookup[slot], port_id, flow_id)) {
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
    int32_t port_id,
    FlowId flow_id,
    Entity entity)
{
    int32_t slot = hashTagLookupKey(port_id, flow_id);

    while (tagLookup[slot].port_id != -1 &&
           !tagLookupMatches(tagLookup[slot], port_id, flow_id)) {
        slot = (slot + 1) & tagLookupMask();
    }

    tagLookup[slot].port_id = port_id;
    tagLookup[slot].flow_id = flow_id;
    tagLookup[slot].entity = entity;
}

MADRONA_NO_INLINE void Sim::removeTagLookup(int32_t port_id, FlowId flow_id)
{
    int32_t slot = findTagLookupSlot(port_id, flow_id);
    if (slot < 0) {
        return;
    }

    int32_t hole = slot;
    int32_t next = (hole + 1) & tagLookupMask();

    while (tagLookup[next].port_id != -1) {
        int32_t home =
            hashTagLookupKey(tagLookup[next].port_id, tagLookup[next].flow_id);
        if (probeDistance(home, next) > probeDistance(home, hole)) {
            tagLookup[hole] = tagLookup[next];
            hole = next;
        }
        next = (next + 1) & tagLookupMask();
    }

    tagLookup[hole] = TagLookupEntry {};
}

MADRONA_NO_INLINE Entity Sim::findTag(
    Context &ctx, int32_t port_id, FlowId flow_id) const
{
    if (port_id < 0 || port_id >= numPorts) {
        return Entity::none();
    }

    int32_t slot = findTagLookupSlot(port_id, flow_id);
    if (slot >= 0) {
        Entity tag_e = tagLookup[slot].entity;
        if (tag_e != Entity::none()) {
            const FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            if (tag.port_id == port_id && tag.flow_id == flow_id) {
                return tag_e;
            }
        }
    }

    const PortTagList &ptl = portTagLists[port_id];
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

MADRONA_NO_INLINE void Sim::removeFlowDef(FlowId flow_id)
{
    int32_t slot = findFlowDefSlot(flow_id);
    if (slot < 0 || slot >= numFlowDefs) {
        return;
    }

    int32_t lookup_idx = flowLookupIndex(flow_id);
    if (lookup_idx >= 0) {
        flowDefSlotLookup[lookup_idx] = -1;
    }

    int32_t last_slot = numFlowDefs - 1;
    if (slot != last_slot) {
        flowDefs[slot] = flowDefs[last_slot];
        int32_t moved_lookup_idx = flowLookupIndex(flowDefs[slot].id);
        if (moved_lookup_idx >= 0) {
            flowDefSlotLookup[moved_lookup_idx] = slot;
        }
    }
    numFlowDefs = last_slot;
    flowDefs[last_slot] = FlowDef {};
}

MADRONA_NO_INLINE void Sim::removeFlowRoute(FlowId flow_id)
{
    int32_t slot = findFlowRouteSlot(flow_id);
    if (slot < 0 || slot >= numFlowRoutes) {
        return;
    }

    int32_t lookup_idx = flowLookupIndex(flow_id);
    if (lookup_idx >= 0) {
        flowRouteSlotLookup[lookup_idx] = -1;
    }

    int32_t last_slot = numFlowRoutes - 1;
    if (slot != last_slot) {
        flowRoutes[slot] = flowRoutes[last_slot];
        int32_t moved_lookup_idx = flowLookupIndex(flowRoutes[slot].flow_id);
        if (moved_lookup_idx >= 0) {
            flowRouteSlotLookup[moved_lookup_idx] = slot;
        }
    }
    numFlowRoutes = last_slot;
    flowRoutes[last_slot] = FlowRouteState {};
}

MADRONA_NO_INLINE void Sim::recordFlowCompletion(
    FlowId flow_id, Time end_time)
{
    int32_t slot = findFlowCompletionSlot(flow_id);
    if (slot >= 0 && slot < numFlowCompletions) {
        if (end_time > flowCompletions[slot].record.end_time) {
            flowCompletions[slot].record.end_time = end_time;
        }
        removeFlowRoute(flow_id);
        removeFlowDef(flow_id);
        return;
    }

    if (numFlowCompletions >= MAX_FLOW_COMPLETIONS) {
        removeFlowRoute(flow_id);
        removeFlowDef(flow_id);
        return;
    }

    const FlowDef *flow = getFlowDef(flow_id);
    if (flow != nullptr) {
        slot = numFlowCompletions++;
        FlowCompletionEntry &entry = flowCompletions[slot];
        entry.flow_id = flow_id;
        entry.record = FlowCompletionRecord {
            .flow_id = flow_id,
            .src_node = flow->src_node,
            .dst_node = flow->dst_node,
            .size = flow->size,
            .start_time = flow->start_time,
            .end_time = end_time,
            .priority = flow->priority,
        };
        int32_t lookup_idx = flowLookupIndex(flow_id);
        if (lookup_idx >= 0) {
            flowCompletionSlotLookup[lookup_idx] = slot;
        }
        removeFlowRoute(flow_id);
        removeFlowDef(flow_id);
        return;
    }

    removeFlowRoute(flow_id);
}

}
