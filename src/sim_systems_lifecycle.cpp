#include "sim.hpp"
#include "sim_debug.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

MADRONA_NO_INLINE bool Sim::destroyTagCollectCleanupEvent(
    Context &ctx,
    Entity tag_entity,
    bool propagate_cleanup,
    Time logical_now,
    DelayedEvent &out_ev)
{
    if (tag_entity == Entity::none()) {
        return false;
    }

    FlowTagState tag = ctx.get<FlowTagState>(tag_entity);
    Time effective_now = logical_now >= 0.0 ? logical_now : now;
    bool has_cleanup_ev = false;

    if (tag.next_port_id < 0) {
        recordFlowCompletion(tag.flow_id, effective_now);
    }

    if (propagate_cleanup && tag.next_port_id >= 0) {
        if (tag.downstream_created != 0) {
            int32_t src_node_slot = findNodeSlot(portToNode[tag.port_id]);
            int32_t dst_node_slot = findNodeSlot(portToNode[tag.next_port_id]);
            Time delay = 0.0;
            if (src_node_slot >= 0 && dst_node_slot >= 0) {
                delay = linkDelays[src_node_slot][dst_node_slot];
                if (delay < 0.0) {
                    delay = 0.0;
                }
            }
            out_ev = DelayedEvent {};
            out_ev.t = effective_now + delay;
            out_ev.type = DelayedEvent::Type::BwUpdate;
            out_ev.bwupd = BwUpdateEv {
                .port_id = tag.next_port_id,
                .flow_id = tag.flow_id,
                .in_bw = 0.0,
            };
            has_cleanup_ev = true;
        } else {
            int32_t cur = tag.port_id;
            int32_t nxt = lookupFlowRouteNext(tag.flow_id, cur);
            while (nxt >= 0) {
                cur = nxt;
                nxt = lookupFlowRouteNext(tag.flow_id, cur);
            }
            if (cur != tag.port_id) {
                recordFlowCompletion(tag.flow_id, effective_now);
            }
        }
    }

    for (int32_t i = 0; i < numTagIndexEntries; i++) {
        if (tagIndex[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numTagIndexEntries; j++) {
                tagIndex[j - 1] = tagIndex[j];
            }
            numTagIndexEntries -= 1;
            break;
        }
    }

    if (tag.port_entity != Entity::none()) {
        int32_t port_id = tag.port_id;
        if (port_id >= 0 && port_id < numPorts) {
            PortTagList &ptl = portTagLists[port_id];
            for (int32_t i = 0; i < ptl.count; i++) {
                if (ptl.tags[i] == tag_entity) {
                    ptl.tags[i] = ptl.tags[ptl.count - 1];
                    ptl.tags[ptl.count - 1] = Entity::none();
                    ptl.count -= 1;
                    break;
                }
            }
        }
    }

    for (int32_t i = 0; i < numIngressTags; i++) {
        if (ingressTags[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numIngressTags; j++) {
                ingressTags[j - 1] = ingressTags[j];
            }
            numIngressTags -= 1;
            i -= 1;
        }
    }

    for (int32_t i = 0; i < numSourceTags; i++) {
        if (sourceTags[i].entity == tag_entity) {
            for (int32_t j = i + 1; j < numSourceTags; j++) {
                sourceTags[j - 1] = sourceTags[j];
            }
            numSourceTags -= 1;
            break;
        }
    }

    if (tag.port_id >= 0 && tag.port_id < numPorts) {
        Entity port_entity = portEntities[tag.port_id];
        if (port_entity != Entity::none()) {
            portDirtyStates[tag.port_id].isDirty = 1;
        }
    }

    ctx.destroyEntity(tag_entity);
    return has_cleanup_ev;
}

MADRONA_NO_INLINE void Sim::destroyTag(Context &ctx,
                                       Entity tag_entity,
                                       bool propagate_cleanup,
                                       Time logical_now)
{
    DelayedEvent ev {};
    if (destroyTagCollectCleanupEvent(
            ctx, tag_entity, propagate_cleanup, logical_now, ev)) {
        pushDelayedEvent(ev);
    }
}

MADRONA_NO_INLINE Entity Sim::createTagOnPort(Context &ctx,
                                              int32_t port_id,
                                              FlowId flow_id,
                                              Bw in_bw,
                                              Bytes size,
                                              bool is_source,
                                              int32_t priority)
{
    if (port_id < 0 || port_id >= numPorts) {
        return Entity::none();
    }

    Entity port_entity = portEntities[port_id];
    if (port_entity == Entity::none()) {
        return Entity::none();
    }

    Entity tag_entity = ctx.makeEntity<FlowTag>();
    FlowTagState tag {};
    tag.port_id = port_id;
    tag.flow_id = flow_id;
    tag.priority = priority;
    tag.in_bw = in_bw;
    tag.out_bw = 0.0;
    tag.prev_out_bw = 0.0;
    tag.backlog = 0.0;
    tag.last_backlog_time = now;
    tag.remaining = size;
    tag.last_remaining_time = now;
    tag.is_source = is_source ? 1 : 0;
    tag.downstream_created = 0;
    tag.next_port_id = lookupFlowRouteNext(flow_id, port_id);
    tag.ingress_port_id = -1;

    if (!is_source) {
        tag.ingress_port_id = lookupFlowIngressPort(flow_id, port_id);
    }

    tag.port_entity = port_entity;
    ctx.get<FlowTagState>(tag_entity) = tag;

    if (numTagIndexEntries < MAX_TAG_INDEX) {
        tagIndex[numTagIndexEntries++] = TagIndexEntry {
            .port_id = port_id,
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    {
        PortTagList &ptl = portTagLists[port_id];
        if (ptl.count < MAX_TAGS_PER_PORT) {
            ptl.tags[ptl.count++] = tag_entity;
        }
    }

    if (tag.ingress_port_id >= 0 && numIngressTags < MAX_INGRESS_TAGS) {
        ingressTags[numIngressTags++] = IngressTagEntry {
            .ingress_port_id = tag.ingress_port_id,
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    if (is_source && numSourceTags < MAX_SOURCE_TAGS) {
        sourceTags[numSourceTags++] = SourceTagEntry {
            .flow_id = flow_id,
            .entity = tag_entity,
        };
    }

    portDirtyStates[port_id].isDirty = 1;
    PortBuffer &port_buf = ctx.get<PortBuffer>(port_entity);
    if (port_buf.last_update_time < now) {
        port_buf.last_update_time = now;
    }

    return tag_entity;
}

}
