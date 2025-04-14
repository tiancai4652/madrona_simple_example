#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

#define SYS_LOG 1

namespace madsimple
{

    void Sim::registerTypes(ECSRegistry &registry, const Config &)
    {
        base::registerTypes(registry);

        registry.registerComponent<Reset>();
        registry.registerComponent<Action>();
        registry.registerComponent<GridPos>();
        registry.registerComponent<Reward>();
        registry.registerComponent<Done>();
        registry.registerComponent<CurStep>();
        registry.registerComponent<ChakraNodesData>();
        registry.registerArchetype<Agent>();

        registry.registerComponent<SimTime>();
        registry.registerArchetype<SimTimeProcessor>();

        // -------------------------------------
        registry.registerComponent<ID>();
        registry.registerComponent<ChakraNodes>();
        registry.registerComponent<HardwareResource>();
        registry.registerComponent<ProcessingCompTask>();
        registry.registerComponent<ProcessingCommTask>();
        registry.registerComponent<Chakra_Nodp_Nodes>();
        registry.registerArchetype<NpuNode>();
        registry.registerComponent<NextProcessTimes>();
        registry.registerArchetype<NextProcessTimeE>();

        // Export tensors for pytorch
        registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
        registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
        registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
        registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
        registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
        registry.exportColumn<Agent, ChakraNodesData>((uint32_t)ExportID::ChakraNodesData);
    }

    // -----------------------------------------------------------------------------------

    // const int INTS_PER_NODE = 44;  // 每个 ChakraNode 占用 44 字节，等于 11 个 int
    // const int MAX_CHAKRA_NODES=9*9999;
    // const int chakra_nodes_data_length=10000000;

    // 将整数数组解析为 ChakraNodes 数组
    int parseChakraNodes(const ChakraNodesData &chakraNodesData, int row, ChakraNode parsedNodes[], int intArrayLength = chakra_nodes_data_length, int maxNodes = MAX_CHAKRA_NODES)
    {

        int validNodes = 0; // 有效节点计数

        for (int i = 0; i < maxNodes; ++i)
        {
            int baseIndex = i * INTS_PER_NODE;
            if (baseIndex + INTS_PER_NODE > intArrayLength)
                break; // 确保不越界

            // 解析单个节点
            for (int j = 0; j < 20; ++j)
            {
                parsedNodes[validNodes].name[j] = chakraNodesData.data[row][baseIndex + j];
            }
            baseIndex += 20;

            parsedNodes[validNodes].type = (NodeType)chakraNodesData.data[row][baseIndex];
            baseIndex += 1;

            parsedNodes[validNodes].id = chakraNodesData.data[row][baseIndex];

            // 初始节点没有id
            if(parsedNodes[validNodes].id==-1)
            {
                parsedNodes[validNodes].id=0;
            }

            baseIndex += 1;

            for (int j = 0; j < 10; ++j)
            {
                parsedNodes[validNodes].data_deps[j] = chakraNodesData.data[row][baseIndex + j];
            }
            baseIndex += 10;

            parsedNodes[validNodes].comm_para = ((uint64_t)chakraNodesData.data[row][baseIndex + 1] << 32) | chakraNodesData.data[row][baseIndex];
            baseIndex += 2;

            parsedNodes[validNodes].comm_size = ((uint64_t)chakraNodesData.data[row][baseIndex + 1] << 32) | chakraNodesData.data[row][baseIndex];
            baseIndex += 2;

            parsedNodes[validNodes].comm_src = ((uint64_t)chakraNodesData.data[row][baseIndex + 1] << 32) | chakraNodesData.data[row][baseIndex];
            baseIndex += 2;

            parsedNodes[validNodes].comm_dst = ((uint64_t)chakraNodesData.data[row][baseIndex + 1] << 32) | chakraNodesData.data[row][baseIndex];
            baseIndex += 2;

            parsedNodes[validNodes].involved_dim_1 = chakraNodesData.data[row][baseIndex];
            parsedNodes[validNodes].involved_dim_2 = chakraNodesData.data[row][baseIndex + 1];
            parsedNodes[validNodes].involved_dim_3 = chakraNodesData.data[row][baseIndex + 2];
            baseIndex += 3;

            parsedNodes[validNodes].durationMicros = chakraNodesData.data[row][baseIndex];
            baseIndex += 1;

            // 检查是否全为 0（表示补全区）
            bool isPadding = true;
            for (int j = i * INTS_PER_NODE; j < i * INTS_PER_NODE + INTS_PER_NODE; ++j)
            {
                if (chakraNodesData.data[row][j] != 0)
                {
                    isPadding = false;
                    break;
                }
            }
            if (isPadding)
            {
                break; // 忽略后续的补全区
            }

            // 完成一个节点解析后递增 validNodes
            validNodes++;
        }
        return validNodes; // 返回有效节点的数量
    }

    // // 定义无效依赖的值
    // const uint32_t INVALID_DEPENDENCY = 4294967295;

    int filterNoDependencyNodes(const ChakraNodes &chakraNodes, ChakraNode filteredNodes[], int maxNodes = CURRENT_EXEC_NODES_MAX)
    {
        
        int count = 0;                       // 无依赖节点计数

        // 遍历所有节点
        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i)
        {
            const ChakraNode &node = chakraNodes.nodes[i];

            // 跳过无效节点
            if (node.type == NodeType::None)
            {
                continue;
            }

            // 检查是否没有依赖
            bool hasDependency = false;
            for (int j = 0; j < 10; ++j)
            {
                if (node.data_deps[j] != INVALID_DEPENDENCY)
                {
                    hasDependency = true;
                    break;
                }
            }

            // 如果没有依赖，添加到结果数组
            if (!hasDependency)
            {
                if (count < maxNodes)
                {
                    filteredNodes[count++] = node;
                }
                else
                {
                    // 超过最大限制，停止添加
                    break;
                }
            }
        }

        return count; // 返回无依赖节点的数量
    }

    void removeNode(ChakraNodes &chakraNodes, uint32_t nodeId) {
       
    
        // 遍历所有节点，找到目标节点并将其标记为无效
        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i) {
            ChakraNode &node = chakraNodes.nodes[i];
    
            // 如果找到目标节点，标记为无效
            if (node.id == nodeId) {
                node.type = NodeType::None; // 标记为无效节点
                for (int j = 0; j < 10; ++j) {
                    node.data_deps[j] = INVALID_DEPENDENCY; // 清空依赖
                }
                node.id = INVALID_DEPENDENCY; // 将节点 ID 设置为无效
                break;
            }
        }
    
        // 遍历所有节点，移除对目标节点的依赖
        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i) {
            ChakraNode &node = chakraNodes.nodes[i];
    
            // 跳过无效节点
            if (node.type == NodeType::None) {
                continue;
            }
    
            // 检查并移除对目标节点的依赖
            for (int j = 0; j < 10; ++j) {
                if (node.data_deps[j] == nodeId) {
                    node.data_deps[j] = INVALID_DEPENDENCY; // 设置为无效依赖
                }
            }
        }
    }

    // network interface
    inline uint64_t getCurrentTime(Engine &ctx)
    {
        return ctx.get<SimTime>(ctx.data().timer_entity).time;
    }

    // network interface
    inline bool isExistedFlow(Engine &ctx)
    {
        return false;
    }

    // network interface
    inline void addSimtime(Engine &ctx,uint64_t t_ns)
    {
        ctx.get<SimTime>(ctx.data().timer_entity).time+=t_ns;
    }

    inline void procssTime(Engine &ctx,
        SimTime &simTime)
    {
    simTime.time += (1000* 1000000);
    printf("current time : %ld\n", simTime.time);
    }

    uint64_t msToNs(uint64_t milliseconds)
    {
        return milliseconds * 1000000; // 1 ms = 1,000,000 ns
    }

    inline void skipTime_add_time(Engine &ctx,uint64_t t_relative_ns)
    {
        for (size_t i = 0; i < MAX_CHAKRA_NODES; i++)
        {
            if(ctx.get<NextProcessTimes>(ctx.data().timer_entity).times_abs[i] != 0)
            {
                ctx.get<NextProcessTimes>(ctx.data().timer_entity).times_abs[i]=getCurrentTime(ctx)+t_relative_ns;
                break;
            }
        }
    }


    // 自定义排序函数：从大到小冒泡排序
    void customSortDescending(uint64_t*array, size_t length) {
        for (size_t i = 0; i < length - 1; ++i) {
            for (size_t j = 0; j < length - 1 - i; ++j) {
                if (array[j] < array[j + 1]) { // 从大到小比较
                    // 交换两个元素
                    uint64_t temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                }
            }
        }
    }



    inline uint64_t skipTime_sort_time_rMin(Engine &ctx)
    {
        // 找到第一个等于 0 的元素的位置
        size_t zeroIndex = MAX_CHAKRA_NODES; // 默认没有找到 0
        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i) {
            if (ctx.get<NextProcessTimes>(ctx.data().timer_entity).times_abs[i] == 0) {
                zeroIndex = i;
                break;
            }
        }

        // 对第一个 0 之前的元素进行从大到小排序
        if (zeroIndex > 0 && zeroIndex < MAX_CHAKRA_NODES) {
            uint64_t *array = ctx.get<NextProcessTimes>(ctx.data().timer_entity).times_abs;
            customSortDescending(array, zeroIndex);
        }
        
        return ctx.get<NextProcessTimes>(ctx.data().timer_entity).times_abs[zeroIndex];
    }

    inline void skipTime_remove_time(Engine &ctx,uint64_t x_ns)
    {
        // 找到第一个等于 0 的元素的位置
        size_t zeroIndex = MAX_CHAKRA_NODES; // 默认没有找到 0
        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i) {
            if (ctx.get<NextProcessTimes>(ctx.data().timer_entity).times_abs[i] != 0) {
                ctx.get<NextProcessTimes>(ctx.data().timer_entity).times_abs[i]-=x_ns;
            }
            else
            {
                break;
            }
        }
    }



    inline void init(Engine &ctx,
                     Action &action,
                     Reset &reset,
                     GridPos &grid_pos,
                     Reward &reward,
                     Done &done,
                     CurStep &episode_step,
                     ChakraNodesData &chakra_nodes_data)
    {

        printf("inside:\n");
        // printf("chakra_nodes_data[0]%d:\n",chakra_nodes_data.data[0][0]);
        // printf("chakra_nodes_data[1]%d:\n",chakra_nodes_data.data[1][0]);

        printf("init npus.\n");
        // 获取行数和列数
        size_t rows = sizeof(chakra_nodes_data.data) / sizeof(chakra_nodes_data.data[0]);       // 总大小 / 单行大小
        size_t cols = sizeof(chakra_nodes_data.data[0]) / sizeof(chakra_nodes_data.data[0][0]); // 单行大小 / 单个元素大小
        size_t npu_nums = rows;
        size_t npu_data_num = cols;
        printf("sizeof(chakra_nodes_data.data):%d\n",sizeof(chakra_nodes_data.data));
        printf("sizeof(chakra_nodes_data.data[0]:%d\n",sizeof(chakra_nodes_data.data[0]));
        printf("npu_nums:%d\n", npu_nums);
        printf("npu_data_num:%d\n", npu_data_num);

        for (int32_t i = 0; i < npu_nums; i++)
        {
            Entity npuNode = ctx.makeEntity<NpuNode>();
            ctx.get<ID>(npuNode).value = i;
            int nodeCount = parseChakraNodes(chakra_nodes_data, i, ctx.get<ChakraNodes>(npuNode).nodes);
            ctx.get<HardwareResource>(npuNode).comp_ocupy = false;
            ctx.get<HardwareResource>(npuNode).comm_ocupy = false;
            ctx.get<HardwareResource>(npuNode).one_task_finish = true;
            ctx.get<ProcessingCompTask>(npuNode).is_none = true;
            printf("npu %d: turn %d nodes.\n", i, nodeCount);
            ctx.data().chakra_nodes_entities[i] = npuNode;
        }

        ctx.destroyEntity(ctx.data().init_entity);
        printf("init npus over.\n");
    }

    inline void processNode(Engine &ctx,
                            ID &id,
                            ChakraNodes &chakraNodes,
                            HardwareResource &hardwareResource,
                            ProcessingCompTask &processingCompTask,
                            ProcessingCommTask &processingCommTask)
    {
        // printf("exec sys: precess_node .");
        // fifter no dp nodes
        if (hardwareResource.one_task_finish && (!hardwareResource.comm_ocupy || !hardwareResource.comp_ocupy))
        {
            // printf("get current execute nodes:\n");
            ChakraNode current_exec_nodes[CURRENT_EXEC_NODES_MAX];
            int count = filterNoDependencyNodes(chakraNodes, current_exec_nodes);
            printf("npu id:%d has %d exec nodes.\n", id.value, count);
            for (size_t i = 0; i < count; i++)
            {
                printf("node id : %d\n",current_exec_nodes[i].id);

            }

            // process no np nodes
            for (size_t i = 0; i < count; i++)
            {
                ChakraNode node = current_exec_nodes[i];
                // printf("node.type:%d\n",node.type);
                switch (node.type)
                {
                    // None=0,
                    // COMP_NODE = 1,
                    // COMM_SEND_NODE = 2,
                    // COMM_RECV_NODE = 3,
                    // COMM_COLL_NODE = 4
                case NodeType::COMP_NODE:
                    if (!hardwareResource.comp_ocupy)
                    {
                        processingCompTask.time_finish_ns = getCurrentTime(ctx) + msToNs(node.durationMicros);
                        printf("processingCompTask.time_finish_ns: %ld\n",processingCompTask.time_finish_ns);
                        processingCompTask.is_none = false;
                        processingCompTask.node_id = node.id;

                        // set flag
                        hardwareResource.comp_ocupy=true;

                        // set skip time - add time.
                        skipTime_add_time(ctx,processingCompTask.time_finish_ns);
                    }
                    break;
                case NodeType::COMM_SEND_NODE:
                    if (!hardwareResource.comm_ocupy)
                    {
                        // test
                        processingCommTask.time_finish_ns = getCurrentTime(ctx) + msToNs(10);
                        printf("processingCommTask.time_finish_ns: %ld\n",processingCommTask.time_finish_ns);
                        processingCommTask.is_none = false;
                        processingCommTask.node_id = node.id;
                        // set flag
                        hardwareResource.comp_ocupy=true;
                    }
                    break;
                case NodeType::COMM_RECV_NODE:
                    if (!hardwareResource.comm_ocupy)
                    {// test
                        processingCommTask.time_finish_ns = getCurrentTime(ctx) + msToNs(10);
                        printf("processingCommTask.time_finish_ns: %ld\n",processingCommTask.time_finish_ns);
                        processingCommTask.is_none = false;
                        processingCommTask.node_id = node.id;
                        // set flag
                        hardwareResource.comp_ocupy=true;
                    }
                    break;
                case NodeType::COMM_COLL_NODE:
                    if (!hardwareResource.comm_ocupy)
                    {// test
                        processingCommTask.time_finish_ns = getCurrentTime(ctx) + msToNs(10);
                        printf("processingCommTask.time_finish_ns: %ld\n",processingCommTask.time_finish_ns);
                        processingCommTask.is_none = false;
                        processingCommTask.node_id = node.id;
                        // set flag
                        hardwareResource.comp_ocupy=true;
                    }
                    break;
                default:
                    break;
                }
            }

            // set flag
            hardwareResource.one_task_finish = false;
        }

        // process comp
        if (!processingCompTask.is_none && (processingCompTask.time_finish_ns <= getCurrentTime(ctx)))
        {
            printf("processingCompTask over.\n");
            // release node.
            removeNode(chakraNodes,processingCompTask.node_id);
            printf("remode node : %d\n",processingCompTask.node_id);
            // reset flag.
            hardwareResource.one_task_finish=true;
            processingCompTask.is_none = true;
            hardwareResource.comp_ocupy=false;


        }
        
        // printf("getCurrentTime(ctx): %d\n",getCurrentTime(ctx));
        // process comm
        if (!processingCommTask.is_none && (processingCommTask.time_finish_ns <= getCurrentTime(ctx)))
        {
            printf("processingCompTask over.\n");
            // release node.
            removeNode(chakraNodes,processingCommTask.node_id);
            printf("remode node : %d\n",processingCommTask.node_id);
            // reset flag.
            hardwareResource.one_task_finish=true;
            processingCommTask.is_none = true;
            hardwareResource.comm_ocupy=false;
        }
    
        // check skip time
        if(!isExistedFlow(ctx))
        {

        }
    }

    // network logic


    uint16_t frame_skiptime=0;
    inline void checkSkipTime(Engine &ctx,NextProcessTimes &t)
    {
        frame_skiptime++;
        if(frame_skiptime/CHECK_SKIPTIME_INTERVAL_PER_FRAME==0)
        {
            frame_skiptime=0;

            if(!isExistedFlow(ctx))
            {
                uint64_t min_time=skipTime_sort_time_rMin(ctx);
                if(min_time!=0)
                {
                    addSimtime(ctx,min_time);
                    skipTime_remove_time(ctx,min_time);
                }
            }
        }
    }

    void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                         const Config &)
    {
        TaskGraphBuilder &builder = taskgraph_mgr.init(0);
        auto sys_init = builder.addToGraph<ParallelForNode<Engine, init,
                                                           Action, Reset, GridPos, Reward, Done, CurStep, ChakraNodesData>>({});
        auto sys_process_node = builder.addToGraph<ParallelForNode<Engine, processNode,
                                                                   ID, ChakraNodes, HardwareResource, ProcessingCompTask,ProcessingCommTask>>({sys_init});
        auto sys_skip_time= builder.addToGraph<ParallelForNode<Engine,checkSkipTime, NextProcessTimes>>({sys_process_node});
        
        auto sys_process_time = builder.addToGraph<ParallelForNode<Engine, procssTime,
                                                                   SimTime>>({sys_skip_time});
    }

    Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
        : WorldBase(ctx),
          episodeMgr(init.episodeMgr),
          grid(init.grid),
          maxEpisodeLength(cfg.maxEpisodeLength)
    {
        Entity agent = ctx.makeEntity<Agent>();
        ctx.get<Action>(agent) = Action::None;
        ctx.get<GridPos>(agent) = GridPos{
            grid->startY,
            grid->startX,
        };
        ctx.get<Reward>(agent).r = 0.f;
        ctx.get<Done>(agent).episodeDone = 0.f;
        ctx.get<CurStep>(agent).step = 0;
        ctx.data().init_entity = agent;
        // printf("1");

        // sim time entity
        Entity sp = ctx.makeEntity<SimTimeProcessor>();
        ctx.get<SimTime>(sp).time = 0;
        ctx.data().timer_entity = sp;

        // next process time entity
        Entity nextProcessTimeE = ctx.makeEntity<NextProcessTimeE>();
        for (size_t i = 0; i < MAX_CHAKRA_NODES; i++)
        {
            ctx.get<NextProcessTimes>(nextProcessTimeE).times_abs[i] = 0;
        }
        ctx.data().next_process_time_entity = nextProcessTimeE;
        
        
        
    }

    MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
