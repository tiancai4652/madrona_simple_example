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

MADRONA_NO_INLINE Entity Sim::findTag(int32_t port_id, FlowId flow_id) const
{
    for (int32_t i = 0; i < numTagIndexEntries; i++) {
        if (tagIndex[i].port_id == port_id && tagIndex[i].flow_id == flow_id) {
            return tagIndex[i].entity;
        }
    }

    return Entity::none();
}

MADRONA_NO_INLINE void Sim::removeFlowDef(FlowId flow_id)
{
    for (int32_t i = 0; i < numFlowDefs; i++) {
        if (flowDefs[i].id != flow_id) {
            continue;
        }
        for (int32_t j = i + 1; j < numFlowDefs; j++) {
            flowDefs[j - 1] = flowDefs[j];
        }
        numFlowDefs -= 1;
        flowDefs[numFlowDefs] = FlowDef {};
        return;
    }
}

MADRONA_NO_INLINE void Sim::removeFlowRoute(FlowId flow_id)
{
    for (int32_t i = 0; i < numFlowRoutes; i++) {
        if (flowRoutes[i].flow_id != flow_id) {
            continue;
        }
        for (int32_t j = i + 1; j < numFlowRoutes; j++) {
            flowRoutes[j - 1] = flowRoutes[j];
        }
        numFlowRoutes -= 1;
        flowRoutes[numFlowRoutes] = FlowRouteState {};
        return;
    }
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

    for (int32_t i = 0; i < numFlowDefs; i++) {
        if (flowDefs[i].id == flow_id) {
            FlowCompletionEntry &entry = flowCompletions[numFlowCompletions++];
            entry.flow_id = flow_id;
            entry.record = FlowCompletionRecord {
                .flow_id = flow_id,
                .src_node = flowDefs[i].src_node,
                .dst_node = flowDefs[i].dst_node,
                .size = flowDefs[i].size,
                .start_time = flowDefs[i].start_time,
                .end_time = end_time,
                .priority = flowDefs[i].priority,
            };
            removeFlowRoute(flow_id);
            removeFlowDef(flow_id);
            return;
        }
    }

    removeFlowRoute(flow_id);
}

}
