#include "sim.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

MADRONA_NO_INLINE void initPortCore(
    Context &ctx,
    Entity port_entity,
    int32_t port_id,
    NodeId node_id,
    int32_t port_idx,
    Bw port_bw)
{
    ctx.get<PortState>(port_entity) = PortState {
        .port_id = port_id,
        .node_id = node_id,
        .port_idx = port_idx,
        .port_bw = port_bw,
        .connected = 0,
        .next_port_id = -1,
    };
    ctx.get<PortBuffer>(port_entity) = PortBuffer {};
    ctx.get<DirtyPort>(port_entity) = DirtyPort {};
}

MADRONA_NO_INLINE void initPortPfc(
    Sim &sim,
    Context &ctx,
    Entity port_entity,
    int32_t port_id)
{
    (void)port_id;
    PortPfcConfig pfc_cfg {};
    if (sim.enablePfc != 0) {
        pfc_cfg.pfc_enabled = 1;
        for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
            pfc_cfg.xoff[i] = sim.pfcXoffThreshold;
            pfc_cfg.xon[i] = sim.pfcXonThreshold;
        }
    }
    ctx.get<PortPfcConfig>(port_entity) = pfc_cfg;
    ctx.get<PortPfcState>(port_entity) = PortPfcState {};
}

MADRONA_NO_INLINE void initPortScratch(
    Context &ctx,
    Entity port_entity)
{
    ctx.get<PortCachedHints>(port_entity) = PortCachedHints {};
    ctx.get<PortDrainHint>(port_entity) = PortDrainHint {};
    ctx.get<PortTimers>(port_entity) = PortTimers {
        .backlog_drain = Sim::timerInactiveSentinel(),
        .pfc_pause = Sim::timerInactiveSentinel(),
        .pfc_resume = Sim::timerInactiveSentinel(),
    };
    ctx.get<PortTraceLast>(port_entity) = PortTraceLast {};
    ctx.get<PortFinishedSourceList>(port_entity) = PortFinishedSourceList {};
    ctx.get<PortDelayedQueue>(port_entity) = PortDelayedQueue {};
}

MADRONA_NO_INLINE void initPortQueues(
    Context &ctx,
    Entity port_entity)
{
    ctx.get<PortCleanup>(port_entity) = PortCleanup {};
    ctx.get<PortOutbox>(port_entity) = PortOutbox {};
    ctx.get<PortTagLookup>(port_entity) = PortTagLookup {};
    ctx.get<PortTagList>(port_entity) = PortTagList {};
    ctx.get<PortSourceTagList>(port_entity) = PortSourceTagList {};
    ctx.get<PortInbox>(port_entity) = PortInbox {};
    ctx.get<PortCreateList>(port_entity) = PortCreateList {};
    PortTagPool pool {};
    pool.free_count = INITIAL_TAGS_PER_PORT;
    pool.allocated_count = INITIAL_TAGS_PER_PORT;
    for (int32_t i = 0; i < INITIAL_TAGS_PER_PORT; i++) {
        Entity tag_entity = ctx.makeEntity<FlowTag>();
        ctx.get<FlowTagState>(tag_entity) = FlowTagState {};
        ctx.get<FlowTagProgress>(tag_entity) = FlowTagProgress {};
        pool.free_tags[i] = tag_entity;
    }
    ctx.get<PortTagPool>(port_entity) = pool;
    ctx.get<PortIngressLinkList>(port_entity) = PortIngressLinkList {};
    ctx.get<PortIngressUnlinkList>(port_entity) = PortIngressUnlinkList {};
    ctx.get<PortDirtyMarkList>(port_entity) = PortDirtyMarkList {};
    ctx.get<PortCompletionList>(port_entity) = PortCompletionList {};
    ctx.get<IngressTagList>(port_entity) = IngressTagList {};
}

} // namespace

int32_t Sim::createPort(Engine &ctx, NodeId node_id, int32_t port_idx, Bw port_bw)
{
    Entity port_entity = ctx.makeEntity<Port>();

    int32_t port_id = nextPortID++;
    portEntities[port_id] = port_entity;
    portToNode[port_id] = node_id;
    numPorts = nextPortID;

    initPortCore(ctx, port_entity, port_id, node_id, port_idx, port_bw);
    initPortPfc(*this, ctx, port_entity, port_id);
    initPortScratch(ctx, port_entity);
    initPortQueues(ctx, port_entity);

    return port_id;
}

}
