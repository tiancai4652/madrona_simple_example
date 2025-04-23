#pragma once

#include <madrona/components.hpp>

namespace madsimple
{

    enum class ExportID : uint32_t
    {
        Reset,
        Action,
        GridPos,
        Reward,
        Done,
        ChakraNodesData,
        NumExports,
    };

    struct Reset
    {
        int32_t resetNow;
    };

    enum class Action : int32_t
    {
        Up = 0,
        Down = 1,
        Left = 2,
        Right = 3,
        None,
    };

    struct GridPos
    {
        int32_t y;
        int32_t x;
    };

    struct Reward
    {
        float r;
    };

    struct Done
    {
        float episodeDone;
    };

    struct CurStep
    {
        uint32_t step;
    };

    struct ChakraNodesData
    {
        uint32_t data[1][10000000];
    };

    struct Agent : public madrona::Archetype<
                       Reset,
                       Action,
                       GridPos,
                       Reward,
                       Done,
                       CurStep,
                       ChakraNodesData>
    {
    };

    struct SimTime
    {
        int64_t time;
    };

    struct SimTimeProcessor : public madrona::Archetype<
                                  SimTime>
    {
    };
// ----------------------------------------

// 每个 ChakraNode 占用 44 字节，等于 11 个 int
#define INTS_PER_NODE 44

#define MAX_CHAKRA_NODES 9 * 9999
#define MAX_CHAKRA_NODP_NODES 99
#define chakra_nodes_data_length 10000000
// 定义无效依赖的值
#define INVALID_DEPENDENCY 4294967295

// 当前无依赖节点的最大数量
#define CURRENT_EXEC_NODES_MAX 10

// 每隔x帧检测一次skip time
#define CHECK_SKIPTIME_INTERVAL_PER_FRAME 100

// 每个npu的最多流数
#define MAX_FLOW_PER_NPU 9999

// 每个comm节点的最大通讯量
#define MAX_FLOW_NUM_PER_COMM_NODE 999

// #define MAX_FLOW_NUM_ALL_COMM_NODE 9999

    struct NpuID
    {
        uint32_t value;
    };

    struct NodeID
    {
        uint32_t value;
    };

    enum class NodeType : int32_t
    {
        None = 0,
        COMP_NODE = 1,
        COMM_SEND_NODE = 2,
        COMM_RECV_NODE = 3,
        COMM_COLL_NODE = 4
    };

    enum CollectiveCommType : int32_t
    {
        ALL_REDUCE = 0,
        REDUCE = 1,
        ALL_GATHER = 2,
        GATHER = 3,
        SCATTER = 4,
        BROADCAST = 5,
        ALL_TO_ALL = 6,
        REDUCE_SCATTER = 7,
        REDUCE_SCATTER_BLOCK = 8,
        BARRIER = 9
    };

    enum CommImplementationType : int32_t
    {
        Ring = 0
    };

    enum class AttributeKey : int32_t
    {
        comm_para = 1,
        comm_size = 2,
        comm_src = 3,
        comm_dst = 4,
        involved_dim = 5
    };

    struct CommModel
    {
        CommImplementationType all_reduce_implementation;
        CommImplementationType all_gather_implementation;
        CommImplementationType reduce_scatter_implementation;
        CommImplementationType all_to_all_implementation;
    };

    struct SysConfig : public madrona::Archetype<
                           CommModel>
    {
    };

    // struct CommParams {
    //     uint64_t comm_size;
    //     uint64_t comm_src;
    //     uint64_t comm_dst;
    //     uint32_t durationMicros;
    //     CollectiveCommType coll_comm_type;
    // };

    enum TaskState : int32_t
    {
        INIT = 0,
        START = 1,
        FINISH = 2
    };

    struct SysFlow
    {
        int id;
        uint64_t comm_size;
        uint64_t comm_src;
        uint64_t comm_dst;
        uint32_t durationMicros;

        // 流执行顺序id
        uint32_t exec_index;
        // 是否执行完
        TaskState state;
        // 收端完成 or 发端完成
        bool is_send;

        SysFlow()
            : id(-1), // 初始化为 0
              comm_size(0),
              comm_src(0),
              comm_dst(0),
              durationMicros(0),
              exec_index(0),
              state(TaskState::INIT),
              is_send(true) // 初始化为 true
        {
        }
    };

    struct TaskFlows
    {
        SysFlow flows[MAX_FLOW_NUM_PER_COMM_NODE];

        void updateFlows(const SysFlow flows_finish[], int flows_finish_size)
        {
            for (int i = 0; i < flows_finish_size; ++i)
            {
                const SysFlow &finishedFlow = flows_finish[i];
                for (int j = 0; j < MAX_FLOW_NUM_PER_COMM_NODE; ++j)
                {
                    if (flows[j].id == finishedFlow.id)
                    { // 查找 id 相同的元素
                        flows[j].state = TaskState::FINISH;
                        flows[j].is_send = finishedFlow.is_send;               // 赋值 is_send
                        flows[j].durationMicros = finishedFlow.durationMicros; // 赋值 durationMicros
                        break;                                                 // 找到匹配项后，跳出内层循环
                    }
                }
            }
        }

        // 检查是否所有的任务都完成了
        bool areAllTasksDone() const
        {
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; ++i)
            {
                if (flows[i].state == TaskState::INIT)
                {
                    continue;
                }

                if (flows[i].state == TaskState::START)
                {
                    return false;
                }
            }
            return true;
        }
    };

    struct ChakraNode
    {
        uint32_t name[20];
        NodeType type;
        uint32_t id;
        uint32_t data_deps[10];
        uint64_t comm_para;
        uint64_t comm_size;
        uint64_t comm_src;
        uint64_t comm_dst;
        bool involved_dim_1;
        bool involved_dim_2;
        bool involved_dim_3;
        uint32_t durationMicros;

        // 自定义赋值操作符
        ChakraNode &operator=(const ChakraNode &other)
        {
            if (this == &other)
                return *this; // 防止自赋值

            // 复制每个成员变量
            for (int i = 0; i < 20; ++i)
            {
                name[i] = other.name[i];
            }
            type = other.type;
            id = other.id;
            for (int i = 0; i < 10; ++i)
            {
                data_deps[i] = other.data_deps[i];
            }
            comm_para = other.comm_para;
            comm_size = other.comm_size;
            comm_src = other.comm_src;
            comm_dst = other.comm_dst;
            involved_dim_1 = other.involved_dim_1;
            involved_dim_2 = other.involved_dim_2;
            involved_dim_3 = other.involved_dim_3;
            durationMicros = other.durationMicros;

            return *this;
        }
    };

    struct ChakraNodes
    {
        ChakraNode nodes[MAX_CHAKRA_NODES];
    };

    struct Chakra_Nodp_Nodes
    {
        ChakraNode nodes[MAX_CHAKRA_NODP_NODES];
    };

    struct HardwareResource
    {
        bool comp_ocupy;
        // bool comm_ocupy;
        bool one_task_finish;

        HardwareResource()
            : comp_ocupy(false),    // 初始化为 0
              one_task_finish(true) // 初始化为 true
        {
        }
    };

    struct NextProcessTimes
    {
        uint64_t times_abs[MAX_CHAKRA_NODES];
    };

    struct NextProcessTimeE : public madrona::Archetype<
                                  NextProcessTimes>
    {
    };

    struct ProcessingCompTask
    {
        int64_t time_finish_ns;
        TaskState state;
        int32_t node_id;
        // 默认构造函数
        ProcessingCompTask()
            : time_finish_ns(0),      // 初始化为 0
              state(TaskState::INIT), // 初始化为 true
              node_id(-1)             // 初始化为 -1 (表示无效节点)
        {
        }
    };

    struct ProcessingCommTask
    {
        int64_t time_finish_ns;
        TaskState state;
        int32_t node_id;
        int32_t flow_count;

        // 默认构造函数
        ProcessingCommTask()
            : time_finish_ns(0),      // 初始化为 0
              state(TaskState::INIT), // 初始化为 true
              node_id(-1),            // 初始化为 -1 (表示无效节点)
              flow_count(0)
        {
        }
    };

    struct ProcessingCommTasks
    {
        ProcessingCommTask tasks[MAX_FLOW_PER_NPU];
        // 默认构造函数
        ProcessingCommTasks()
        {
            for (int i = 0; i < MAX_FLOW_PER_NPU; ++i)
            {
                tasks[i] = ProcessingCommTask(); // 初始化每个任务
            }
        }

        // 统计所有任务的 flow_count 的总和
        int getTotalFlowCount() const
        {
            int totalFlowCount = 0;
            for (int i = 0; i < MAX_FLOW_PER_NPU; ++i)
            {
                if (tasks[i].state != TaskState::INIT)
                { // 只统计有效任务
                    totalFlowCount += tasks[i].flow_count;
                }
            }
            return totalFlowCount;
        }

        // 判断是否存在指定节点 ID 的方法
        bool containsNodeId(int32_t node_id) const
        {
            for (int i = 0; i < MAX_FLOW_PER_NPU; ++i)
            {
                // 任务状态可能是start也可能是finish，还没来得及处理
                if (tasks[i].state != TaskState::INIT && tasks[i].node_id == node_id)
                {
                    return true; // 找到匹配的任务
                }
            }
            return false; // 未找到匹配的任务
        }

        void setFinish(int32_t node_id, int64_t time_finish_ns)
        {
            for (int i = 0; i < MAX_FLOW_PER_NPU; ++i)
            {      
                if (tasks[i].state == TaskState::START && tasks[i].node_id == node_id)
                {
                    printf("node id %d -> setFinish\n",node_id);
                    tasks[i].state = TaskState::FINISH;
                    tasks[i].time_finish_ns = time_finish_ns;
                    break;
                }
            }
        }

        // 添加任务的方法
        bool addTask(const ProcessingCommTask &new_task)
        {
            for (int i = 0; i < MAX_FLOW_PER_NPU; ++i)
            {
                if (tasks[i].state == TaskState::INIT)
                {                                      // 找到第一个 node_id == -1 的位置
                    tasks[i] = new_task;               // 放入新任务
                    tasks[i].state = TaskState::START; // 标记任务为有效
                    return true;                       // 添加成功
                }
            }
            return false; // 没有空闲位置，添加失败
        }

        // 判断是否至少存在一个 is_none == false 的节点
        bool has_task() const
        {
            for (int i = 0; i < MAX_FLOW_PER_NPU; ++i)
            {
                if (tasks[i].state == TaskState::START || tasks[i].state == TaskState::FINISH)
                { // 如果找到一个 is_none == false 的节点
                    return true;
                }
            }
            return false; // 如果所有节点的 is_none == true
        }

        // 查找 time_finish_ns <= t 的所有任务，并将这些任务的 is_none 设置为 true
        int dequeueTasksByTime(int64_t t, ProcessingCommTask result[], int max_result_size)
        {
            int count = 0;
            for (int i = 0; i < MAX_FLOW_PER_NPU && count < max_result_size; ++i)
            {
                if (tasks[i].state == TaskState::FINISH && tasks[i].time_finish_ns <= t)
                {
                    result[count++] = tasks[i];       // 将任务放入结果数组
                    tasks[i].state = TaskState::INIT; // 标记任务为无效（出队）
                    // tasks[i].node_id = -1;         // 重置 node_id
                    // tasks[i].time_finish_ns = 0;   // 重置 time_finish_ns
                }
            }
            return count; // 返回找到的任务个数
        }
    };

    struct NpuNode : public madrona::Archetype<
                         NpuID,
                         ChakraNodes,
                         HardwareResource,
                         ProcessingCompTask,
                         ProcessingCommTasks,
                         Chakra_Nodp_Nodes
                        >
    {
    };

    // ID &id, CollectiveCommType &collective_comm_type, CommParams &comm_params,TaskFlows &taskFlows
    struct ProcessComm_E : public madrona::Archetype<
    NpuID,NodeID,
                               // CollectiveCommType,
                               // CommParams,
                               TaskFlows
                              >
    {
    };

    // ------------------------------------------

}
