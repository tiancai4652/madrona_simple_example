#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

MADRONA_NO_INLINE int32_t Sim::findSourceTagIndex(FlowId flow_id) const
{
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
    if (flowLookupSpan > 0) {
        int64_t lookup_idx = (int64_t)flow_id - (int64_t)flowLookupBase;
        if (lookup_idx >= 0 && lookup_idx < flowLookupSpan) {
            return flowDefSlotLookup[(int32_t)lookup_idx];
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
    if (flowLookupSpan > 0) {
        int64_t lookup_idx = (int64_t)flow_id - (int64_t)flowLookupBase;
        if (lookup_idx >= 0 && lookup_idx < flowLookupSpan) {
            return flowRouteSlotLookup[(int32_t)lookup_idx];
        }
    }

    for (int32_t i = 0; i < numFlowRoutes; i++) {
        if (flowRoutes[i].flow_id == flow_id) {
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

MADRONA_NO_INLINE Entity Sim::findTag(
    Context &ctx, int32_t port_id, FlowId flow_id) const
{
    if (port_id < 0 || port_id >= numPorts) {
        return Entity::none();
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

    for (int32_t j = slot + 1; j < numFlowDefs; j++) {
        flowDefs[j - 1] = flowDefs[j];
        if (flowLookupSpan > 0) {
            flowDefSlotLookup[flowDefs[j - 1].id - flowLookupBase] = j - 1;
        }
    }
    if (flowLookupSpan > 0) {
        int64_t lookup_idx = (int64_t)flow_id - (int64_t)flowLookupBase;
        if (lookup_idx >= 0 && lookup_idx < flowLookupSpan) {
            flowDefSlotLookup[(int32_t)lookup_idx] = -1;
        }
    }
    numFlowDefs -= 1;
    flowDefs[numFlowDefs] = FlowDef {};
}

MADRONA_NO_INLINE void Sim::removeFlowRoute(FlowId flow_id)
{
    int32_t slot = findFlowRouteSlot(flow_id);
    if (slot < 0 || slot >= numFlowRoutes) {
        return;
    }

    for (int32_t j = slot + 1; j < numFlowRoutes; j++) {
        flowRoutes[j - 1] = flowRoutes[j];
        if (flowLookupSpan > 0) {
            flowRouteSlotLookup[flowRoutes[j - 1].flow_id - flowLookupBase] =
                j - 1;
        }
    }
    if (flowLookupSpan > 0) {
        int64_t lookup_idx = (int64_t)flow_id - (int64_t)flowLookupBase;
        if (lookup_idx >= 0 && lookup_idx < flowLookupSpan) {
            flowRouteSlotLookup[(int32_t)lookup_idx] = -1;
        }
    }
    numFlowRoutes -= 1;
    flowRoutes[numFlowRoutes] = FlowRouteState {};
}

MADRONA_NO_INLINE void Sim::recordFlowCompletion(
    FlowId flow_id, Time end_time)
{
    for (int32_t i = 0; i < numFlowCompletions; i++) {
        if (flowCompletions[i].flow_id == flow_id) {
            if (end_time > flowCompletions[i].record.end_time) {
                flowCompletions[i].record.end_time = end_time;
            }
            removeFlowRoute(flow_id);
            removeFlowDef(flow_id);
            return;
        }
    }

    if (numFlowCompletions >= MAX_FLOW_COMPLETIONS) {
        removeFlowRoute(flow_id);
        removeFlowDef(flow_id);
        return;
    }

    const FlowDef *flow = getFlowDef(flow_id);
    if (flow != nullptr) {
        FlowCompletionEntry &entry = flowCompletions[numFlowCompletions++];
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
        removeFlowRoute(flow_id);
        removeFlowDef(flow_id);
        return;
    }

    removeFlowRoute(flow_id);
}

}
