#pragma once

#include <madrona/components.hpp>

namespace madsimple {

enum class ExportID : uint32_t {
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    ChakraNodesData,
    NumExports,
};

struct Reset {
    int32_t resetNow;
};

enum class Action : int32_t {
    Up    = 0,
    Down  = 1,
    Left  = 2,
    Right = 3,
    None,
};

struct GridPos {
    int32_t y;
    int32_t x;
};

struct Reward {
    float r;
};

struct Done {
    float episodeDone;
};

struct CurStep {
    uint32_t step;
};

struct ChakraNodesData {
    uint32_t data[1][10000000];
};

struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep,
    ChakraNodesData
> {};

struct  SimTime
{
    int64_t time;
};

struct SimTimeProcessor : public madrona::Archetype<
    SimTime
> {};
// ----------------------------------------

#define INTS_PER_NODE  44  // 每个 ChakraNode 占用 44 字节，等于 11 个 int
#define MAX_CHAKRA_NODES 9*9999
#define MAX_CHAKRA_NODP_NODES 99
#define chakra_nodes_data_length 10000000
// 定义无效依赖的值
#define INVALID_DEPENDENCY 4294967295

#define CURRENT_EXEC_NODES_MAX 10

#define CHECK_SKIPTIME_INTERVAL_PER_FRAME 100

#define MAX_FLOW_PER_NPU 9999

struct ID {
    uint32_t value;
};

enum class NodeType : int32_t {
    None=0,
    COMP_NODE = 1,
    COMM_SEND_NODE = 2,
    COMM_RECV_NODE = 3,
    COMM_COLL_NODE = 4
};

enum class AttributeKey: int32_t {
    comm_para = 1,
    comm_size = 2,
    comm_src = 3,
    comm_dst = 4,
    involved_dim = 5
};

struct SysFlow {
    uint32_t id;
    uint64_t comm_size;
    uint64_t comm_src;
    uint64_t comm_dst;
    uint32_t durationMicros;
};

struct ChakraNode {
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
     ChakraNode &operator=(const ChakraNode &other) {
        if (this == &other) return *this; // 防止自赋值

        // 复制每个成员变量
        for (int i = 0; i < 20; ++i) {
            name[i] = other.name[i];
        }
        type = other.type;
        id = other.id;
        for (int i = 0; i < 10; ++i) {
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

struct ChakraNodes{
    ChakraNode nodes[MAX_CHAKRA_NODES];
};

struct Chakra_Nodp_Nodes{
    ChakraNode nodes[MAX_CHAKRA_NODP_NODES];
};

struct HardwareResource{
    bool comp_ocupy;
    // bool comm_ocupy;
    bool one_task_finish;

    HardwareResource()
        : comp_ocupy(false), // 初始化为 0
        one_task_finish(true)     // 初始化为 true
    {}

};

struct NextProcessTimes{
    uint64_t times_abs[MAX_CHAKRA_NODES];
};

struct NextProcessTimeE : public madrona::Archetype<
NextProcessTimes
> {};


struct ProcessingCompTask{
    int64_t time_finish_ns;
    bool is_none;
    int32_t node_id;
    // 默认构造函数
    ProcessingCompTask()
        : time_finish_ns(0), // 初始化为 0
          is_none(true),     // 初始化为 true
          node_id(-1)        // 初始化为 -1 (表示无效节点)
    {}
};

struct ProcessingCommTask{
    int64_t time_finish_ns;
    bool is_none;
    int32_t node_id;

    // 默认构造函数
    ProcessingCommTask()
        : time_finish_ns(0), // 初始化为 0
          is_none(true),     // 初始化为 true
          node_id(-1)        // 初始化为 -1 (表示无效节点)
    {}
};

struct ProcessingCommTasks{
    ProcessingCommTask tasks[MAX_FLOW_PER_NPU];
    // 默认构造函数
    ProcessingCommTasks() {
        for (int i = 0; i < MAX_FLOW_PER_NPU; ++i) {
            tasks[i] = ProcessingCommTask(); // 初始化每个任务
        }
    }

    // 判断是否存在指定节点 ID 的方法
    bool containsNodeId(int32_t node_id) const {
        for (int i = 0; i < MAX_FLOW_PER_NPU; ++i) {
            if (!tasks[i].is_none && tasks[i].node_id == node_id) {
                return true; // 找到匹配的任务
            }
        }
        return false; // 未找到匹配的任务
    }

    // 添加任务的方法
    bool addTask(const ProcessingCommTask &new_task) {
        for (int i = 0; i < MAX_FLOW_PER_NPU; ++i) {
            if (tasks[i].is_none == true ){ // 找到第一个 node_id == -1 的位置
                tasks[i] = new_task; // 放入新任务
                tasks[i].is_none = false; // 标记任务为有效
                return true; // 添加成功
            }
        }
        return false; // 没有空闲位置，添加失败
    }

    // 判断是否至少存在一个 is_none == false 的节点
    bool is_none() const {
        for (int i = 0; i < MAX_FLOW_PER_NPU; ++i) {
            if (!tasks[i].is_none) { // 如果找到一个 is_none == false 的节点
                return false;
            }
        }
        return true; // 如果所有节点的 is_none == true
    }

    // 查找 time_finish_ns <= t 的所有任务，并将这些任务的 is_none 设置为 true
    int dequeueTasksByTime(int64_t t, ProcessingCommTask result[], int max_result_size) {
        int count = 0;
        for (int i = 0; i < MAX_FLOW_PER_NPU && count < max_result_size; ++i) {
            if (!tasks[i].is_none && tasks[i].time_finish_ns <= t) {
                result[count++] = tasks[i];     // 将任务放入结果数组
                tasks[i].is_none = true;       // 标记任务为无效（出队）
                // tasks[i].node_id = -1;         // 重置 node_id
                // tasks[i].time_finish_ns = 0;   // 重置 time_finish_ns
            }
        }
        return count; // 返回找到的任务个数
    }
};

struct NpuNode : public madrona::Archetype<
    ID,
    ChakraNodes,
    HardwareResource,
    ProcessingCompTask,
    ProcessingCommTasks,
    Chakra_Nodp_Nodes
> {};

// ------------------------------------------

}
