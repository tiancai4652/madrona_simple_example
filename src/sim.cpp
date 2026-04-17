#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

void scheduleStepSystem(Engine &ctx, SimDriver &driver)
{
    Sim &sim = ctx.data();
    sim.systemLogStep += 1;
    driver.tick += 1;
    sim.schedulePendingFlows();
}

void deliverStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.deliverEvents();
}

void arrivalStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flowArrivalSystem(ctx);
}

void bwUpdateStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.bwUpdateIngressSystem(ctx);
}

void pfcPropagateStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    if (sim.enablePfc != 0) {
        sim.pfcPropagateSystem(ctx);
    }
}

void portAllocStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.portBandwidthAllocSystem(ctx, 0.0);
}

void pfcDetectStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.pfcThresholdDetectSystem(ctx);
}

void downstreamEmitStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.downstreamEmitSystem(ctx);
}

void clearDirtyStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.clearDirtyPorts(ctx);
}

void chooseDTStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.nextDT = sim.chooseDT();
    if (sim.nextDT < 1e-9) {
        sim.nextDT = 0.001;
    }
}

void bufferUpdateStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.bufferUpdateSystem(ctx, sim.nextDT);
}

void flowProgressStepSystem(Engine &ctx, SimDriver &)
{
    Sim &sim = ctx.data();
    sim.flowProgressAndCleanupSystem(ctx, sim.nextDT);
    sim.now += sim.nextDT;
}

}

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    registry.registerComponent<SimDriver>();

    registry.registerComponent<DirtyPort>();
    registry.registerComponent<PortState>();
    registry.registerComponent<FlowTagState>();
    registry.registerComponent<PortBuffer>();
    registry.registerComponent<PortPfcConfig>();
    registry.registerComponent<PortPfcState>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<SimDriverArch>();
    registry.registerArchetype<Port>();
    registry.registerArchetype<FlowTag>();

    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);

    auto n0 = builder.addToGraph<ParallelForNode<Engine,
        scheduleStepSystem, SimDriver>>({});
    auto n1 = builder.addToGraph<ParallelForNode<Engine,
        deliverStepSystem, SimDriver>>({n0});
    auto n2 = builder.addToGraph<ParallelForNode<Engine,
        arrivalStepSystem, SimDriver>>({n1});
    auto n3 = builder.addToGraph<ParallelForNode<Engine,
        bwUpdateStepSystem, SimDriver>>({n2});
    auto n4 = builder.addToGraph<ParallelForNode<Engine,
        pfcPropagateStepSystem, SimDriver>>({n3});
    auto n5 = builder.addToGraph<ParallelForNode<Engine,
        portAllocStepSystem, SimDriver>>({n4});
    auto n6 = builder.addToGraph<ParallelForNode<Engine,
        pfcDetectStepSystem, SimDriver>>({n5});
    auto n7 = builder.addToGraph<ParallelForNode<Engine,
        downstreamEmitStepSystem, SimDriver>>({n6});
    auto n8 = builder.addToGraph<ParallelForNode<Engine,
        clearDirtyStepSystem, SimDriver>>({n7});
    auto n9 = builder.addToGraph<ParallelForNode<Engine,
        chooseDTStepSystem, SimDriver>>({n8});
    auto n10 = builder.addToGraph<ParallelForNode<Engine,
        bufferUpdateStepSystem, SimDriver>>({n9});
    auto n11 = builder.addToGraph<ParallelForNode<Engine,
        flowProgressStepSystem, SimDriver>>({n10});

#ifdef MADRONA_GPU_MODE
    auto recycle_entities = builder.addToGraph<RecycleEntitiesNode>({n11});
    (void)recycle_entities;
#else
    (void)n11;
#endif
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      network(init.network),
      maxEpisodeLength(cfg.maxEpisodeLength),
      now(0.0),
      nextPortID(0),
      numTopoNodes(0),
      numTopoLinks(0),
      numPorts(0),
      numFlowDefs(0),
      numPendingFlows(0),
      numFlowRoutes(0)
{
    resetNetworkState();
    enableBuffer = cfg.enable_buffer;
    enablePfc = cfg.enable_pfc;
    pfcEgress = cfg.pfc_egress;
    defaultLinkDelay = cfg.default_link_delay;
    propagationInterval = cfg.propagation_interval;
    pfcXoffThreshold = cfg.pfc_xoff_threshold;
    pfcXonThreshold = cfg.pfc_xon_threshold;
    dtMin = cfg.dt_min;
    qosMode = cfg.qos_mode;
    for (int32_t i = 0; i < PFC_MAX_PRIORITY; i++) {
        priorWeights[i] = cfg.prior_weights[i];
    }

    Entity agent = ctx.makeEntity<Agent>();
    ctx.get<Reset>(agent) = Reset { .resetNow = 0 };
    ctx.get<Action>(agent) = Action::None;
    ctx.get<GridPos>(agent) = GridPos { .y = 0, .x = 0 };
    ctx.get<Reward>(agent) = Reward { .r = 0.f };
    ctx.get<Done>(agent) = Done { .episodeDone = 0.f };
    ctx.get<CurStep>(agent) = CurStep { .step = 0 };

    Entity driver = ctx.makeEntity<SimDriverArch>();
    ctx.get<SimDriver>(driver) = SimDriver { .tick = 0 };

    loadTopo(ctx);
    loadFlow(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
