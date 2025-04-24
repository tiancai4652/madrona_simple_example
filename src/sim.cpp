#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

#define SYS_LOG true

#define SYS_CHECK true

#define ENABLE_TEST true

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
        registry.registerComponent<NpuID>();
        registry.registerComponent<NodeID>();
        registry.registerComponent<ChakraNodes>();
        registry.registerComponent<HardwareResource>();
        registry.registerComponent<ProcessingCompTask>();
        registry.registerComponent<ProcessingCommTasks>();
        registry.registerComponent<Chakra_Nodp_Nodes>();
        registry.registerArchetype<NpuNode>();

        registry.registerComponent<NextProcessTimes>();
        registry.registerArchetype<NextProcessTimeE>();

        registry.registerComponent<CommModel>();
        registry.registerArchetype<SysConfig>();

        registry.registerComponent<TaskFlows>();
        registry.registerArchetype<ProcessComm_E>();

        // Export tensors for pytorch
        registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
        registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
        registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
        registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
        registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
        registry.exportColumn<Agent, ChakraNodesData>((uint32_t)ExportID::ChakraNodesData);
    }

    // -----------------------------------------------------------------------------------

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
            if (parsedNodes[validNodes].id == -1)
            {
                parsedNodes[validNodes].id = 0;
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

    int filterNoDependencyNodes(const ChakraNodes &chakraNodes, ChakraNode filteredNodes[], int maxNodes = CURRENT_EXEC_NODES_MAX)
    {

        int count = 0; // 无依赖节点计数

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

    void removeNode(ChakraNodes &chakraNodes, uint32_t nodeId)
    {

        // 遍历所有节点，找到目标节点并将其标记为无效
        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i)
        {
            ChakraNode &node = chakraNodes.nodes[i];

            // 如果找到目标节点，标记为无效
            if (node.id == nodeId)
            {
                node.type = NodeType::None; // 标记为无效节点
                for (int j = 0; j < 10; ++j)
                {
                    node.data_deps[j] = INVALID_DEPENDENCY; // 清空依赖
                }
                node.id = INVALID_DEPENDENCY; // 将节点 ID 设置为无效
                break;
            }
        }

        // 遍历所有节点，移除对目标节点的依赖
        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i)
        {
            ChakraNode &node = chakraNodes.nodes[i];

            // 跳过无效节点
            if (node.type == NodeType::None)
            {
                continue;
            }

            // 检查并移除对目标节点的依赖
            for (int j = 0; j < 10; ++j)
            {
                if (node.data_deps[j] == nodeId)
                {
                    node.data_deps[j] = INVALID_DEPENDENCY; // 设置为无效依赖
                }
            }
        }
    }

    // network interface
    // 获取当前仿真时间
    // func
    inline uint64_t getCurrentTime(Engine &ctx)
    {
        return ctx.get<SimTime>(ctx.data().timer_entity).time;
    }

    // network interface
    // 指示当前是否存在流发送
    // func
    inline bool isExistedFlow(Engine &ctx)
    {
        return false;
    }

    // network interface
    // 增加仿真时间：使所有组件增加仿真时间
    // func
    inline void addSimtime(Engine &ctx, uint64_t t_ns)
    {
        ctx.get<SimTime>(ctx.data().timer_entity).time += t_ns;
    }

    // network interface
    // 设置流：
    // func
    inline void setFlow(Engine &ctx, uint64_t comm_src, uint64_t comm_dst, uint64_t comm_size, uint32_t flow_id)
    {
        printf("set flow id %d: %d->%d %d, \n", flow_id, comm_src, comm_dst, comm_size);
    }

    // network interface
    // 检测/获取流完成事件：返回流完成数量，同时把流完成事件赋值flows_finish
    // func
    inline uint32_t checkFlowFinish(Engine &ctx, uint32_t npu_id, SysFlow flows_finish[])
    {
        return 0;
    }

    inline void procssTime(Engine &ctx,
                           SimTime &simTime)
    {
        // simTime.time += (1000 * 1000000);
        simTime.time += (1000);
        printf("current time : %ld\n", simTime.time);
    }

    uint64_t msToNs(uint64_t milliseconds)
    {
        return milliseconds * 1000000; // 1 ms = 1,000,000 ns
    }

    inline void skipTime_add_time(Engine &ctx, uint64_t t_relative_ns)
    {
        for (size_t i = 0; i < MAX_CHAKRA_NODES; i++)
        {
            if (ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs[i] != 0)
            {
                ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs[i] = getCurrentTime(ctx) + t_relative_ns;
                break;
            }
        }
    }

    // 自定义排序函数：从大到小冒泡排序
    void customSortDescending(uint64_t *array, size_t length)
    {
        for (size_t i = 0; i < length - 1; ++i)
        {
            for (size_t j = 0; j < length - 1 - i; ++j)
            {
                if (array[j] < array[j + 1])
                { // 从大到小比较
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

        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i)
        {

            if (ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs[i] == 0)
            {

                zeroIndex = i;
                break;
            }
        }

        // 对第一个 0 之前的元素进行从大到小排序
        if (zeroIndex > 0 && zeroIndex < MAX_CHAKRA_NODES)
        {
            uint64_t *array = ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs;
            customSortDescending(array, zeroIndex);
        }

        return ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs[zeroIndex];
    }

    inline void skipTime_remove_time(Engine &ctx, uint64_t x_ns)
    {
        // 找到第一个等于 0 的元素的位置
        size_t zeroIndex = MAX_CHAKRA_NODES; // 默认没有找到 0
        for (size_t i = 0; i < MAX_CHAKRA_NODES; ++i)
        {
            if (ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs[i] != 0)
            {
                ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs[i] -= x_ns;
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
        printf("init npus.\n");
        // 获取行数和列数
        size_t rows = sizeof(chakra_nodes_data.data) / sizeof(chakra_nodes_data.data[0]);       // 总大小 / 单行大小
        size_t cols = sizeof(chakra_nodes_data.data[0]) / sizeof(chakra_nodes_data.data[0][0]); // 单行大小 / 单个元素大小
        size_t npu_nums = rows;
        size_t npu_data_num = cols;
        printf("sizeof(chakra_nodes_data.data):%d\n", sizeof(chakra_nodes_data.data));
        printf("sizeof(chakra_nodes_data.data[0]:%d\n", sizeof(chakra_nodes_data.data[0]));
        printf("npu_nums:%d\n", npu_nums);
        printf("npu_data_num:%d\n", npu_data_num);

        for (int32_t i = 0; i < npu_nums; i++)
        {
            Entity npuNode = ctx.makeEntity<NpuNode>();
            ctx.get<NpuID>(npuNode).value = i;
            int nodeCount = parseChakraNodes(chakra_nodes_data, i, ctx.get<ChakraNodes>(npuNode).nodes);
            ctx.get<HardwareResource>(npuNode) = HardwareResource();
            ctx.get<ProcessingCompTask>(npuNode) = ProcessingCompTask();
            ctx.get<ProcessingCommTasks>(npuNode) = ProcessingCommTasks();
            // ctx.get<Entity>(npuNode) = npuNode;

            printf("npu %d: turn %d nodes.\n", i, nodeCount);
            ctx.data().chakra_nodes_entities[i] = npuNode;
        }

        ctx.destroyEntity(ctx.data().init_entity);
        printf("init npus over.\n");
    }

    inline void processNpuNodes(Engine &ctx,
                                NpuID &id,
                                ChakraNodes &chakraNodes,
                                HardwareResource &hardwareResource,
                                ProcessingCompTask &processingCompTask,
                                ProcessingCommTasks &processingCommTasks)
    {
        if (SYS_LOG && id.value == 0)
        {
            printf("### sys1 ### : exec processNpuNodes.\n");
        }

        // append nodes
        if (hardwareResource.one_task_finish)
        {
            ChakraNode current_exec_nodes[CURRENT_EXEC_NODES_MAX];
            int count = filterNoDependencyNodes(chakraNodes, current_exec_nodes);
            printf("total no dp node:%d\n", count);
            // process no np nodes
            for (size_t i = 0; i < count; i++)
            {

                ChakraNode node = current_exec_nodes[i];
                // 会重复查找，已经在处理时，跳过
                if (processingCommTasks.containsNodeId(node.id))
                {
                    continue;
                }
                if (hardwareResource.comp_ocupy && node.id == processingCompTask.node_id)
                {
                    continue;
                }
                if (SYS_LOG && id.value == 0)
                {
                    printf("----process node %d:----\n", current_exec_nodes[i].id);
                }
                switch (node.type)
                {
                case NodeType::COMP_NODE:
                {
                    if (!hardwareResource.comp_ocupy)
                    {
                        processingCompTask.time_finish_ns = getCurrentTime(ctx) + msToNs(node.durationMicros);
                        if (SYS_LOG && id.value == 0)
                        {
                            printf("processingCompTask.time_finish_ns: %ld\n", processingCompTask.time_finish_ns);
                        }
                        processingCompTask.state = TaskState::START;
                        processingCompTask.node_id = node.id;

                        // set flag
                        hardwareResource.comp_ocupy = true;

                        // set skip time - add time.
                        skipTime_add_time(ctx, processingCompTask.time_finish_ns);
                    }
                    else
                    {
                        if (SYS_LOG && id.value == 0)
                        {
                            printf("no comp hardware!");
                        }
                    }
                    break;
                }
                case NodeType::COMM_SEND_NODE:
                {

                    ProcessingCommTask processingCommTask = ProcessingCommTask();
                    if (SYS_LOG && id.value == 0)
                    {
                        printf("processingCommTask send %d -> %d .\n", node.comm_src, node.comm_dst);
                    }
                    processingCommTask.state = TaskState::START;
                    processingCommTask.node_id = node.id;
                    processingCommTask.flow_count = 1;
                    processingCommTasks.addTask(processingCommTask);

                    // create comm entity
                    Entity process_e = ctx.makeEntity<ProcessComm_E>();
                    ctx.get<NpuID>(process_e).value = id.value;
                    ctx.get<NodeID>(process_e).value = node.id;
                    uint32_t flow_id = processingCommTasks.getTotalFlowCount() + 1;
                    ctx.get<TaskFlows>(process_e).flows[0] = SysFlow();
                    ctx.get<TaskFlows>(process_e).flows[0].id = flow_id;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_size = node.comm_size;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_src = node.comm_src;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_dst = node.comm_dst;
                    ctx.get<TaskFlows>(process_e).flows[0].state = TaskState::START;
                    ctx.get<TaskFlows>(process_e).flows[0].is_send = true;

                    ctx.data().node_flows_exec_entity[id.value][node.id]= process_e;
                    // setFlow(Engine &ctx, uint64_t comm_src, uint64_t comm_dst, uint64_t comm_size, uint32_t flow_id)
                    setFlow(ctx, node.comm_src, node.comm_dst, node.comm_size, flow_id);

                    break;
                }
                case NodeType::COMM_RECV_NODE:
                {
                    ProcessingCommTask processingCommTask = ProcessingCommTask();
                    if (SYS_LOG && id.value == 0)
                    {
                        printf("processingCommTask recv %d -> %d .\n", node.comm_src, node.comm_dst);
                    }
                    processingCommTask.state = TaskState::START;
                    processingCommTask.node_id = node.id;
                    processingCommTask.flow_count = 1;
                    processingCommTasks.addTask(processingCommTask);

                    // create comm entity
                    Entity process_e = ctx.makeEntity<ProcessComm_E>();
                    ctx.get<NpuID>(process_e).value = id.value;
                    ctx.get<NodeID>(process_e).value = node.id;
                    uint32_t flow_id = processingCommTasks.getTotalFlowCount() + 1;
                    ctx.get<TaskFlows>(process_e).flows[0] = SysFlow();
                    ctx.get<TaskFlows>(process_e).flows[0].id = flow_id;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_size = node.comm_size;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_src = node.comm_src;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_dst = node.comm_dst;
                    ctx.get<TaskFlows>(process_e).flows[0].state = TaskState::START;
                    ctx.get<TaskFlows>(process_e).flows[0].is_send = false;

                    ctx.data().node_flows_exec_entity[id.value][node.id]= process_e;
                    // setFlow(Engine &ctx, uint64_t comm_src, uint64_t comm_dst, uint64_t comm_size, uint32_t flow_id)
                    setFlow(ctx, node.comm_src, node.comm_dst, node.comm_size, flow_id);

                    break;
                }
                case NodeType::COMM_COLL_NODE:
                {
                    ProcessingCommTask processingCommTask = ProcessingCommTask();
                    if (SYS_LOG && id.value == 0)
                    {
                        printf("processingCommTask: coll .\n");
                    }
                    processingCommTask.state = TaskState::START;
                    processingCommTask.node_id = node.id;
                    processingCommTask.flow_count = 1;
                    processingCommTasks.addTask(processingCommTask);

                    // create comm entity
                    Entity process_e = ctx.makeEntity<ProcessComm_E>();
                    ctx.get<NpuID>(process_e).value = id.value;
                    ctx.get<NodeID>(process_e).value = node.id;
                    uint32_t flow_id = processingCommTasks.getTotalFlowCount() + 1;
                    ctx.get<TaskFlows>(process_e).flows[0] = SysFlow();
                    ctx.get<TaskFlows>(process_e).flows[0].id = flow_id;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_size = 100000;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_src = 0;
                    ctx.get<TaskFlows>(process_e).flows[0].comm_dst = 1;
                    ctx.get<TaskFlows>(process_e).flows[0].state = TaskState::START;
                    ctx.get<TaskFlows>(process_e).flows[0].is_send = true;


                    ctx.data().node_flows_exec_entity[id.value][node.id]= process_e;


                    // setFlow(Engine &ctx, uint64_t comm_src, uint64_t comm_dst, uint64_t comm_size, uint32_t flow_id)
                    setFlow(ctx, 0, 1, 100000, flow_id);
                    break;
                }
                default:
                    break;
                }
            }

            // set flag
            hardwareResource.one_task_finish = false;
        }
    }

    inline void processCommCheckFlow(Engine &ctx, NpuID &npu_id, NodeID &node_id, TaskFlows &taskFlows)
    {
        if (SYS_LOG && npu_id.value == 0)
        {
            printf("### sys2 ### : exec processCommCheckFlow.\n");
        }

        SysFlow flows_finish[MAX_FLOW_NUM_PER_COMM_NODE];
        uint32_t flow_finish_count = checkFlowFinish(ctx, npu_id.value, flows_finish);
        if (ENABLE_TEST)
        {
            for (int i = 0; i < 20; ++i)
            {
                if (taskFlows.flows[i].state == TaskState::START)
                {
                    taskFlows.flows[i].state = TaskState::FINISH;
                    printf("test : flow id %d -> set flow finish.\n", taskFlows.flows[i].id);
                }
            }
        }

        if (flow_finish_count > 0)
        {
            taskFlows.updateFlows(flows_finish, flow_finish_count);
        }

        if (taskFlows.areAllTasksDone())
        {
            printf("node id %d -> taskFlows.areAllTasksDone\n", node_id);

            ctx.get<ProcessingCommTasks>(ctx.data().chakra_nodes_entities[npu_id.value]).setFinish(node_id.value, getCurrentTime(ctx));
        
            ctx.destroyEntity(ctx.data().node_flows_exec_entity[npu_id.value][node_id.value]);
        }
    }

    // network logic

    uint16_t frame_skiptime = 0;
    inline void checkSkipTime(Engine &ctx, NextProcessTimes &t)
    {
        if (SYS_LOG)
        {
            printf("### sys4 ### : exec checkSkipTime.\n");
        }

        frame_skiptime++;
        if (frame_skiptime / CHECK_SKIPTIME_INTERVAL_PER_FRAME == 0)
        {
            frame_skiptime = 0;
            if (!isExistedFlow(ctx))
            {
                uint64_t min_time = skipTime_sort_time_rMin(ctx);
                if (min_time != 0)
                {
                    addSimtime(ctx, min_time - getCurrentTime(ctx));
                    if (SYS_LOG)
                    {
                        printf("skip to time:%d\n", getCurrentTime(ctx));
                    }
                    skipTime_remove_time(ctx, min_time);
                }
            }
        }
    }

    inline void removeNpuNodes(Engine &ctx,
                               NpuID &id,
                               ChakraNodes &chakraNodes,
                               HardwareResource &hardwareResource,
                               ProcessingCompTask &processingCompTask,
                               ProcessingCommTasks &processingCommTasks)
    {
        if (SYS_LOG && id.value == 0)
        {
            printf("### sys3 ### : exec removeNpuNodes.\n");
        }

        if (SYS_LOG && id.value == 0)
        {
            printf("----check comp task:----\n");
        }
        // process comp
        if (processingCompTask.state == TaskState::START && (processingCompTask.time_finish_ns >= getCurrentTime(ctx)))
        {
            // if (SYS_LOG && id.value == 0)
            // {
            //     printf("processingCompTask %d over.\n", processingCompTask.node_id);
            // }
            // release node.
            removeNode(chakraNodes, processingCompTask.node_id);
            if (SYS_LOG && id.value == 0)
            {
                printf("remode comp node : %d\n", processingCompTask.node_id);
            }
            // reset flag.
            hardwareResource.one_task_finish = true;
            processingCompTask.state = TaskState::INIT;
            hardwareResource.comp_ocupy = false;
        }
        if (SYS_LOG && id.value == 0)
        {
            printf("----check comm tasks:----\n");
        }
        // process comm
        if (processingCommTasks.has_task())
        {
            // if (SYS_LOG && id.value == 0)
            // {
            //     printf("node id %d -> processingCommTasks.has_task() .\n", id.value);
            // }

            int64_t t = getCurrentTime(ctx);
            ProcessingCommTask result[MAX_FLOW_PER_NPU];
            int task_count = processingCommTasks.dequeueTasksByTime(t, result, MAX_FLOW_PER_NPU);

            if (SYS_LOG && id.value == 0 && task_count > 0)
            {
                printf("npu id %d -> processingCommTasks over count :%d .\n", id.value, task_count);
            }

            if (task_count > 0)
            {
                for (size_t i = 0; i < task_count; i++)
                {
                    // if (SYS_LOG && id.value == 0)
                    // {
                    //     printf("processingCompTask over.\n");
                    // }
                    // release node.
                    removeNode(chakraNodes, processingCommTasks.tasks[i].node_id);
                    if (SYS_LOG && id.value == 0)
                    {
                        printf("release comm node : %d\n", processingCommTasks.tasks[i].node_id);
                    }
                }
            }

            // reset flag.
            hardwareResource.one_task_finish = true;
        }
    }

    void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                         const Config &)
    {
        TaskGraphBuilder &builder = taskgraph_mgr.init(0);
        auto sys_init = builder.addToGraph<ParallelForNode<Engine, init,
                                                           Action, Reset, GridPos, Reward, Done, CurStep, ChakraNodesData>>({});
        auto sys_process_node = builder.addToGraph<ParallelForNode<Engine, processNpuNodes,
                                                                   NpuID, ChakraNodes, HardwareResource, ProcessingCompTask, ProcessingCommTasks>>({sys_init});
        auto sys_process_comm = builder.addToGraph<ParallelForNode<Engine, processCommCheckFlow,
                                                                   NpuID, NodeID, TaskFlows>>({sys_process_node});
        
        auto sys_remove_node=builder.addToGraph<ParallelForNode<Engine, removeNpuNodes,
        NpuID, ChakraNodes, HardwareResource, ProcessingCompTask, ProcessingCommTasks>>({sys_process_comm});
        
        auto sys_skip_time = builder.addToGraph<ParallelForNode<Engine, checkSkipTime, NextProcessTimes>>({sys_remove_node});

        auto sys_process_time = builder.addToGraph<ParallelForNode<Engine, procssTime,
                                                                   SimTime>>({sys_skip_time});
    }

    inline void SetEntity()
    {
    }

    inline void ReadEntity()
    {
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

        // Sys_config
        Entity sc = ctx.makeEntity<SysConfig>();
        ctx.get<CommModel>(sc).all_reduce_implementation = CommImplementationType::Ring;
        ctx.get<CommModel>(sc).all_gather_implementation = CommImplementationType::Ring;
        ctx.get<CommModel>(sc).reduce_scatter_implementation = CommImplementationType::Ring;
        ctx.get<CommModel>(sc).all_to_all_implementation = CommImplementationType::Ring;
        ctx.data().sys_config_entity = sc;
    }

    MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);
}
