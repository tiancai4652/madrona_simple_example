#include "sim.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

MADRONA_NO_INLINE void initPortCore(
    Sim &sim,
    Context &ctx,
    Entity port_entity,
    int32_t port_id,
    NodeId node_id,
    int32_t port_idx,
    Bw port_bw)
{
    sim.portDirtyStates[port_id].isDirty = 0;
    ctx.get<PortState>(port_entity) = PortState {
        .port_id = port_id,
        .node_id = node_id,
        .port_idx = port_idx,
        .port_bw = port_bw,
        .connected = 0,
        .next_port_id = -1,
    };
    ctx.get<PortBuffer>(port_entity) = PortBuffer {};
}

MADRONA_NO_INLINE void initPortPfc(
    Sim &sim,
    int32_t port_id)
{
    PortPfcConfig pfc_cfg {};
    if (sim.enablePfc != 0) {
        pfc_cfg.pfc_enabled = 1;
        for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
            pfc_cfg.xoff[i] = sim.pfcXoffThreshold;
            pfc_cfg.xon[i] = sim.pfcXonThreshold;
        }
    }
    sim.portPfcConfigs[port_id] = pfc_cfg;
    sim.portPfcStates[port_id] = PortPfcState {};
}

MADRONA_NO_INLINE void initPortScratch(
    Sim &sim,
    int32_t port_id)
{
    sim.portCachedHints[port_id] = PortCachedHints {};
    sim.portDrainHints[port_id] = PortDrainHint {};
    sim.portTraceLasts[port_id] = PortTraceLast {};
}

MADRONA_NO_INLINE void initPortQueues(
    Sim &sim,
    int32_t port_id)
{
    sim.portCleanups[port_id] = PortCleanup {};
    sim.portOutboxes[port_id] = PortOutbox {};
    sim.portTagLists[port_id] = PortTagList {};
    sim.portInboxes[port_id] = PortInbox {};
    sim.portCreateLists[port_id] = PortCreateList {};
    sim.portCompletionLists[port_id] = PortCompletionList {};
}

} // namespace

int32_t Sim::createPort(Engine &ctx, NodeId node_id, int32_t port_idx, Bw port_bw)
{
    Entity port_entity = ctx.makeEntity<Port>();

    int32_t port_id = nextPortID++;
    portEntities[port_id] = port_entity;
    portToNode[port_id] = node_id;
    numPorts = nextPortID;

    initPortCore(*this, ctx, port_entity, port_id, node_id, port_idx, port_bw);
    initPortPfc(*this, port_id);
    initPortScratch(*this, port_id);
    initPortQueues(*this, port_id);

    return port_id;
}

}
