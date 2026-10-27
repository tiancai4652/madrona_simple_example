#include "sim.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

// GPU compile-time note: wholesale `x = T{}` on multi-KB components makes
// NVRTC -O3 emit tens of thousands of unrolled stores (PortBuffer alone is
// ~64KB), costing minutes of compile time per TU. The helpers below use
// zeroStruct() for the bulk zero plus per-element value-init loops that
// restore every array element's exact `T{}` state (NSDMIs like
// PortTagLookupEntry's flow_id=-1 / Entity::none() are honored). Each loop
// carries #pragma unroll 1 so NVRTC keeps it a loop.
// See /app/report/优化gpu编译速度/8*.md.

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
    {
        auto &c = ctx.get<PortBuffer>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
#pragma unroll 1
            for (int32_t ch = 0; ch < MAX_BUFFER_CHUNKS; ch++) {
#pragma unroll 1
                for (int32_t w = 0; w < MAX_CHUNK_WEIGHTS; w++) {
                    c.prior_bufs[i].buf_chunks[ch].weights[w].flow_id = -1;
                }
            }
        }
    }
    zeroStruct(ctx.get<DirtyPort>(port_entity));
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
    zeroStruct(ctx.get<PortPfcState>(port_entity));
}

MADRONA_NO_INLINE void initPortScratch(
    Context &ctx,
    Entity port_entity)
{
    zeroStruct(ctx.get<PortCachedHints>(port_entity));
    zeroStruct(ctx.get<PortDrainHint>(port_entity));
    ctx.get<PortTimers>(port_entity) = PortTimers {
        .backlog_drain = Sim::timerInactiveSentinel(),
        .pfc_pause = Sim::timerInactiveSentinel(),
        .pfc_resume = Sim::timerInactiveSentinel(),
    };
    zeroStruct(ctx.get<PortTraceLast>(port_entity));
    {
        auto &c = ctx.get<PortFinishedSourceList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_CLEANUP; i++) {
            c.tags[i] = Entity {};
        }
    }
    {
        auto &c = ctx.get<PortDelayedQueue>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_DELAYED_EVENTS; i++) {
            c.events[i] = DelayedEvent {};
        }
    }
    {
        // GPU stack relief (2026-10-23): per-port frame scratch. Counts are
        // reset at each phase entry; Entity arrays are none-initialized so
        // any out-of-count read would fail loudly instead of aliasing id 0.
        auto &c = ctx.get<PortFrameScratch>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_TAGS_PER_PORT; i++) {
            c.alloc.tags[i] = Entity::none();
            c.alloc.live_tags[i] = Entity::none();
            c.qos.tags[0][i] = Entity::none();
            c.adv_tags[i] = Entity::none();
            c.adv_pri_tags[0][i] = Entity::none();
        }
    }
}

MADRONA_NO_INLINE void initPortQueues(
    Context &ctx,
    Entity port_entity)
{
    {
        auto &c = ctx.get<PortCleanup>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_CLEANUP; i++) {
            c.tags[i] = Entity {};
            c.propagate[i] = 0;
        }
    }
    {
        auto &c = ctx.get<PortOutbox>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_OUTBOX; i++) {
            c.events[i] = DelayedEvent {};
        }
    }
    {
        auto &c = ctx.get<PortTagLookup>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_TAG_LOOKUP; i++) {
            c.entries[i] = PortTagLookupEntry {};
        }
    }
    {
        auto &c = ctx.get<PortTagList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_TAGS_PER_PORT; i++) {
            c.tags[i] = Entity {};
        }
    }
    {
        auto &c = ctx.get<PortSourceTagList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_TAGS_PER_PORT; i++) {
            c.tags[i] = Entity {};
        }
    }
    {
        auto &c = ctx.get<PortInbox>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_INBOX_ARRIVAL; i++) {
            c.arrivals[i] = FlowArrivalEv {};
        }
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_INBOX_BWUPD; i++) {
            c.bwupds[i] = BwUpdateEv {};
        }
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_INBOX_PFC; i++) {
            c.pfcs[i] = PfcControlEv {};
        }
    }
    {
        auto &c = ctx.get<PortCreateList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_CREATE; i++) {
            c.reqs[i] = PortCreateReq {};
        }
    }
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
    {
        auto &c = ctx.get<PortIngressLinkList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_INGRESS_LINKS; i++) {
            c.reqs[i] = PortIngressLinkReq {};
        }
    }
    {
        auto &c = ctx.get<PortIngressUnlinkList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_INGRESS_UNLINKS; i++) {
            c.reqs[i] = PortIngressUnlinkReq {};
        }
    }
    {
        auto &c = ctx.get<PortDirtyMarkList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_DIRTY_MARKS; i++) {
            c.port_ids[i] = 0;
        }
    }
    {
        auto &c = ctx.get<PortCompletionList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_PORT_COMPLETE; i++) {
            c.reqs[i] = PortCompletionReq {};
        }
    }
    {
        auto &c = ctx.get<IngressTagList>(port_entity);
        zeroStruct(c);
#pragma unroll 1
        for (int32_t i = 0; i < MAX_TAGS_PER_INGRESS; i++) {
            c.tags[i] = Entity {};
        }
    }
}

} // namespace

int32_t Sim::createPort(Engine &ctx, NodeId node_id, int32_t port_idx, Bw port_bw)
{
    Entity port_entity = ctx.makeEntity<Port>();

    int32_t port_id = nextPortID++;
    if (port_id >= MAX_TOPO_PORTS) {
        FATAL("capacity exceeded: MAX_TOPO_PORTS=%d (need port %d) -- raise MAX_TOPO_PORTS",
              (int)MAX_TOPO_PORTS, (int)port_id);
    }
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
