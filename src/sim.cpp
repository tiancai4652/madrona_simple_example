#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

struct StepScheduleNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().systemLogStep += 1;
        ctx.data().schedulePendingFlows();
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepScheduleNode>(deps);
    }
};

struct StepDeliverNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().deliverEvents();
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepDeliverNode>(deps);
    }
};

struct StepArrivalNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().flowArrivalSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepArrivalNode>(deps);
    }
};

struct StepBwUpdateNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().bwUpdateIngressSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepBwUpdateNode>(deps);
    }
};

struct StepPfcPropagateNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        if (ctx.data().enablePfc != 0) {
            ctx.data().pfcPropagateSystem(ctx);
        }
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepPfcPropagateNode>(deps);
    }
};

struct StepPortAllocNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().portBandwidthAllocSystem(ctx, 0.0);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepPortAllocNode>(deps);
    }
};

struct StepPfcDetectNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().pfcThresholdDetectSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepPfcDetectNode>(deps);
    }
};

struct StepDownstreamEmitNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().downstreamEmitSystem(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepDownstreamEmitNode>(deps);
    }
};

struct StepClearDirtyNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().clearDirtyPorts(ctx);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepClearDirtyNode>(deps);
    }
};

struct StepChooseDTNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().nextDT = ctx.data().chooseDT();
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepChooseDTNode>(deps);
    }
};

struct StepBufferUpdateNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().bufferUpdateSystem(ctx, ctx.data().nextDT);
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepBufferUpdateNode>(deps);
    }
};

struct StepFlowProgressNode : public NodeBase {
    void run(Context &ctx_base, TaskGraph &)
    {
        Engine &ctx = (Engine &)ctx_base;
        ctx.data().flowProgressAndCleanupSystem(ctx, ctx.data().nextDT);
        ctx.data().now += ctx.data().nextDT;
    }

    static TaskGraphNodeID addToGraph(StateManager &, TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps)
    {
        return builder.addDefaultNode<StepFlowProgressNode>(deps);
    }
};

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

    registry.registerComponent<DirtyPort>();
    registry.registerComponent<PortState>();
    registry.registerComponent<FlowTagState>();
    registry.registerComponent<PortBuffer>();
    registry.registerComponent<PortPfcConfig>();
    registry.registerComponent<PortPfcState>();

    registry.registerArchetype<Agent>();
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
    TaskGraphNodeID n0 = builder.addToGraph<StepScheduleNode>({});
    TaskGraphNodeID n1 = builder.addToGraph<StepDeliverNode>({n0});
    TaskGraphNodeID n2 = builder.addToGraph<StepArrivalNode>({n1});
    TaskGraphNodeID n3 = builder.addToGraph<StepBwUpdateNode>({n2});
    TaskGraphNodeID n4 = builder.addToGraph<StepPfcPropagateNode>({n3});
    TaskGraphNodeID n5 = builder.addToGraph<StepPortAllocNode>({n4});
    TaskGraphNodeID n6 = builder.addToGraph<StepPfcDetectNode>({n5});
    TaskGraphNodeID n7 = builder.addToGraph<StepDownstreamEmitNode>({n6});
    TaskGraphNodeID n8 = builder.addToGraph<StepClearDirtyNode>({n7});
    TaskGraphNodeID n9 = builder.addToGraph<StepChooseDTNode>({n8});
    TaskGraphNodeID n10 = builder.addToGraph<StepBufferUpdateNode>({n9});
    builder.addToGraph<StepFlowProgressNode>({n10});
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

    loadTopo(ctx);
    loadFlow(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
