#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include "sys_layer/sysComponent.hpp"
#include "sys_layer/Astrasim.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    

    registry.registerArchetype<Agent>();

    // 注册系统层组件和实体
    
    registry.registerComponent<SystemLayerComponent>();
    registry.registerComponent<SystemConfig>();
    registry.registerArchetype<SystemLayerAgent>();


    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
}

inline void tick(Engine &ctx,
                 Action &action,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step)
{
    // const GridState *grid = ctx.data().grid;

    // GridPos new_pos = grid_pos;

    // switch (action) {
    //     case Action::Up: {
    //         new_pos.y += 1;
    //     } break;
    //     case Action::Down: {
    //         new_pos.y -= 1;
    //     } break;
    //     case Action::Left: {
    //         new_pos.x -= 1;
    //     } break;
    //     case Action::Right: {
    //         new_pos.x += 1;
    //     } break;
    //     default: break;
    // }

    // action = Action::None;

    // if (new_pos.x < 0) {
    //     new_pos.x = 0;
    // }

    // if (new_pos.x >= grid->width) {
    //     new_pos.x = grid->width - 1;
    // }

    // if (new_pos.y < 0) {
    //     new_pos.y = 0;
    // }

    // if (new_pos.y >= grid->height) {
    //     new_pos.y = grid->height -1;
    // }

    // {
    //     const Cell &new_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

    //     if ((new_cell.flags & CellFlag::Wall)) {
    //         new_pos = grid_pos;
    //     }
    // }

    // const Cell &cur_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

    // bool episode_done = false;
    // if (reset.resetNow != 0) {
    //     reset.resetNow = 0;
    //     episode_done = true;
    // }

    // if ((cur_cell.flags & CellFlag::End)) {
    //     episode_done = true;
    // }

    // uint32_t cur_step = episode_step.step;

    // if (cur_step == ctx.data().maxEpisodeLength - 1) {
    //     episode_done = true;
    // }

    // if (episode_done) {
    //     done.episodeDone = 1.f;

    //     new_pos = GridPos {
    //         grid->startY,
    //         grid->startX,
    //     };

    //     episode_step.step = 0;
    // } else {
    //     done.episodeDone = 0.f;
    //     episode_step.step = cur_step + 1;
    // }

    // // Commit new position
    // grid_pos = new_pos;
    // reward.r = cur_cell.reward;
}

// 添加一个处理系统层的函数
inline void processSystemLayer(Engine &ctx,
                             SystemLayerComponent &sysComponent,
                             SystemConfig &config)
{
    // 如果系统层指针为空，则初始化它
    if (sysComponent.systemLayerPtr == nullptr) {
        sysComponent.systemLayerPtr = new AstraSim::SystemLayer();
        sysComponent.systemLayerPtr->initialize();
        
        // 应用配置
        sysComponent.systemLayerPtr->setParameter("simulationSpeed", config.simulationSpeed);
        sysComponent.systemLayerPtr->setParameter("accuracy", config.accuracy);
        
        printf("系统层已初始化\n");
    }
    
    // 更新系统层
    float deltaTime = 0.1f;
    sysComponent.systemLayerPtr->update(deltaTime);
    
    // 获取当前模拟结果
    AstraSim::SimulationData result = sysComponent.systemLayerPtr->getSimulationResult();
    
    // 输出一些调试信息
    if (static_cast<int>(result.time * 10) % 10 == 0) {  // 每隔一段时间输出一次
        printf("系统层更新: 时间=%f, 值=%f\n", result.time, result.value);
    }
}



void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);
        // 添加原有的任务节点
    auto tick_sys = builder.addToGraph<ParallelForNode<Engine, tick,
        Action, Reset, GridPos, Reward, Done, CurStep>>({});
    
    // 添加系统层处理任务节点
    builder.addToGraph<ParallelForNode<Engine, processSystemLayer,
        SystemLayerComponent, SystemConfig>>({tick_sys});
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    Entity agent = ctx.makeEntity<Agent>();
    ctx.get<Action>(agent) = Action::None;
    ctx.get<GridPos>(agent) = GridPos {
        grid->startY,
        grid->startX,
    };
    ctx.get<Reward>(agent).r = 0.f;
    ctx.get<Done>(agent).episodeDone = 0.f;
    ctx.get<CurStep>(agent).step = 0;

    // 创建系统层实体
    Entity sysAgent = ctx.makeEntity<SystemLayerAgent>();
    
    // 配置 SystemConfig
    SystemConfig &config = ctx.get<SystemConfig>(sysAgent);
    config.simulationSpeed = 2.0f;
    config.accuracy = 0.98f;
    config.maxIterations = 2000;
    // config.simulationMode = "advanced";
    stringToIntArray( "advanced", config.simulationMode);

    // 初始化 SystemLayerComponent
    SystemLayerComponent &sysComponent = ctx.get<SystemLayerComponent>(sysAgent);
    sysComponent.systemLayerPtr = nullptr;  // 将在 processSystemLayer 中初始化
   
    
    printf("已创建系统层实体\n");
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
