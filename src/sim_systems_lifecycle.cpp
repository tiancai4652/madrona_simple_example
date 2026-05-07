#include "sim.hpp"
#include "sim_debug.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

inline void appendPortSourceTag(PortSourceTagList &source_list,
                                Entity tag_entity)
{
    if (source_list.overflow != 0) {
        return;
    }

    if (source_list.count < MAX_TAGS_PER_PORT) {
        source_list.tags[source_list.count++] = tag_entity;
    } else {
        source_list.overflow = 1;
    }
}

inline void removePortSourceTag(PortSourceTagList &source_list,
                                Entity tag_entity)
{
    if (source_list.overflow != 0) {
        return;
    }

    for (int32_t i = 0; i < source_list.count; i++) {
        if (source_list.tags[i] != tag_entity) {
            continue;
        }

        source_list.tags[i] = source_list.tags[source_list.count - 1];
        source_list.tags[source_list.count - 1] = Entity::none();
        source_list.count -= 1;
        break;
    }
}

inline void removePortFinishedSource(PortFinishedSourceList &finished_list,
                                     Entity tag_entity)
{
    for (int32_t i = 0; i < finished_list.num; i++) {
        if (finished_list.tags[i] != tag_entity) {
            continue;
        }

        finished_list.tags[i] = finished_list.tags[finished_list.num - 1];
        finished_list.tags[finished_list.num - 1] = Entity::none();
        finished_list.num -= 1;
        break;
    }
}

} // namespace

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
        recordFlowCompletion(ctx, tag.flow_id, effective_now);
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
            int32_t nxt = lookupFlowRouteNext(ctx, tag.flow_id, cur);
            while (nxt >= 0) {
                cur = nxt;
                nxt = lookupFlowRouteNext(ctx, tag.flow_id, cur);
            }
            if (cur != tag.port_id) {
                recordFlowCompletion(ctx, tag.flow_id, effective_now);
            }
        }
    }

    if (numTagIndexEntries > 0) {
        numTagIndexEntries -= 1;
    }

    if (tag.port_entity != Entity::none()) {
        int32_t port_id = tag.port_id;
        if (port_id >= 0 && port_id < numPorts) {
            removeTagLookup(ctx.get<PortTagLookup>(tag.port_entity),
                tag.flow_id);
            PortTagList &ptl = ctx.get<PortTagList>(tag.port_entity);
            for (int32_t i = 0; i < ptl.count; i++) {
                if (ptl.tags[i] == tag_entity) {
                    ptl.tags[i] = ptl.tags[ptl.count - 1];
                    ptl.tags[ptl.count - 1] = Entity::none();
                    ptl.count -= 1;
                    break;
                }
            }

            if (tag.is_source != 0) {
                removePortSourceTag(
                    ctx.get<PortSourceTagList>(tag.port_entity), tag_entity);
                removePortFinishedSource(
                    ctx.get<PortFinishedSourceList>(tag.port_entity),
                    tag_entity);
                ctx.get<PortCachedHints>(tag.port_entity).active_finish_t =
                    timerInactiveSentinel();
            }
        }
    }

    if (tag.ingress_port_id >= 0 && tag.ingress_port_id < numPorts) {
        Entity ingress_entity = portEntities[tag.ingress_port_id];
        if (ingress_entity != Entity::none()) {
            IngressTagList &itl = ctx.get<IngressTagList>(ingress_entity);
            for (int32_t i = 0; i < itl.count; i++) {
                if (itl.tags[i] != tag_entity) {
                    continue;
                }
                itl.tags[i] = itl.tags[itl.count - 1];
                itl.tags[itl.count - 1] = Entity::none();
                itl.count -= 1;
                break;
            }
        }
    }

    if (tag.is_source != 0) {
        Entity flow_entity = findFlowMetaEntity(ctx, tag.flow_id);
        if (flow_entity != Entity::none()) {
            FlowRuntimeState &runtime =
                ctx.get<FlowRuntimeState>(flow_entity);
            if (runtime.source_tag_entity == tag_entity) {
                runtime.source_tag_entity = Entity::none();
            }
        }
        if (numSourceTags > 0) {
            numSourceTags -= 1;
        }
    }

    if (tag.port_id >= 0 && tag.port_id < numPorts) {
        Entity port_entity = portEntities[tag.port_id];
        if (port_entity != Entity::none()) {
            ctx.get<DirtyPort>(port_entity).isDirty = 1;
        }
    }

    ctx.get<FlowTagProgress>(tag_entity).pending_source_destroy = 0;

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
        pushDelayedEvent(ctx, ev);
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
    tag.next_port_id = lookupFlowRouteNext(ctx, flow_id, port_id);
    tag.ingress_port_id = -1;

    if (!is_source) {
        tag.ingress_port_id = lookupFlowIngressPort(ctx, flow_id, port_id);
    }

    tag.port_entity = port_entity;
    ctx.get<FlowTagState>(tag_entity) = tag;
    ctx.get<FlowTagProgress>(tag_entity) = FlowTagProgress {};
    insertTagLookup(ctx.get<PortTagLookup>(port_entity), flow_id, tag_entity);

    numTagIndexEntries += 1;

    {
        PortTagList &ptl = ctx.get<PortTagList>(port_entity);
        if (ptl.count < MAX_TAGS_PER_PORT) {
            ptl.tags[ptl.count++] = tag_entity;
        }
    }

    if (is_source) {
        appendPortSourceTag(ctx.get<PortSourceTagList>(port_entity),
            tag_entity);
    }

    if (tag.ingress_port_id >= 0) {
        Entity ingress_entity = portEntities[tag.ingress_port_id];
        if (ingress_entity != Entity::none()) {
            IngressTagList &itl = ctx.get<IngressTagList>(ingress_entity);
            if (itl.count < MAX_TAGS_PER_INGRESS) {
                itl.tags[itl.count++] = tag_entity;
            } else {
                FATAL("IngressTagList overflow");
            }
        }
    }

    if (is_source) {
        numSourceTags += 1;
        Entity flow_entity = findFlowMetaEntity(ctx, flow_id);
        if (flow_entity != Entity::none()) {
            ctx.get<FlowRuntimeState>(flow_entity).source_tag_entity =
                tag_entity;
        }
    }

    ctx.get<DirtyPort>(port_entity).isDirty = 1;
    PortBuffer &port_buf = ctx.get<PortBuffer>(port_entity);
    if (port_buf.last_update_time < now) {
        port_buf.last_update_time = now;
    }

    return tag_entity;
}

}
