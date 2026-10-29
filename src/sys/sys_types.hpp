#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>

#include "inference_types.hpp"

// // ----------- for llm sys -------------------------

// // System configuration macros
// #define ENABLE_TEST true

// Each ChakraNode occupies 51 ints with new 3-int attribute format
#define INTS_PER_NODE 51

#define NPU_NUM 1024


// #define MAX_CHAKRA_NODES_PER_NPU 9 * 9999
// 036-需求x (layer0): 每 rank 10 节点 (6 COMP + 2 OUTCAST + 2 INCAST),
// 取 32 留 3.2x 余量. 派生维度 (npus_chakra_exec_entity/ChakraNodes/
// SharedChakraTemplates/ReadyState 等) 随宏自动跟随.
#define MAX_CHAKRA_NODES_PER_NPU 1024
#define MAX_CONCURRENT_CHAKRA_NODES_PER_NPU 3
#define MAX_CHAKRA_NODP_NODES 5
// 036-需求x (layer0): 实测行宽 1581 ints (=10x51 主表 +51 零哨兵 +4x255 side 区),
// 取 4096 (2 的幂, 16KB/rank).
// 036 只读共享: 4P+4D 共享 256 唯一行 (P 0..127 + D 128..255), 单表仅 4MB.
// 与 python chakra/config.py 同步.
#define CHAKRA_DATA_ROWS 256
#define CHAKRA_NODES_DATA_LENGTH 4096 // aligned with tensor export size in mgr.cpp
// Define invalid dependency value
#define INVALID_DEPENDENCY 2147483646
#define NODE_NAME_LENGTH 20
#define NODE_DATA_DEPS_LENGTH 10
#define NODE_ATTR_PER_LENGTH 3
#define NODE_DURATION_MICROS_LENGTH 1


// Define invalid flow ID value
#define INVALID_FLOW_ID 2147483646

// Data type enumeration for attribute storage
enum class AttributeDataType : uint32_t
{
    BOOL_VAL = 0,   // boolVal
    UINT32_VAL = 1, // uint32Val
    INT64_VAL = 2,  // int64Val
    UINT64_VAL = 3, // uint64Val
    BOOL_LIST = 4   // boolList
};

// Maximum number of current dependency-free nodes
#define CURRENT_EXEC_NODES_MAX 10

// Check skip time every x frames
#define CHECK_SKIPTIME_INTERVAL_PER_FRAME 100

// 018-layer0-GPUfit: layer0 每 NPU 并发 comm 槽位实测峰值<=4
// (2 OUTCAST + 2 INCAST; inference 仅 reset 不占 slot, 见 inference_scheduler.cpp),
// 取 16 留 4x 余量. NpuID.result/node_task_count 同宏自动跟随.
// 超限回退到线性扫描 (语义不变), 溢出由 getNextFreeTask 复用兜底.
#define MAX_COMM_TASK_PER_NPU 512

// Maximum communication volume per comm node
#define MAX_FLOW_NUM_PER_COMM_NODE 128

// Flow ID range interval per node
#define FLOW_ID_MAX_LENGTH 65536

// #define MAX_FLOW_NUM_ALL_COMM_NODE 9999

// 019 扇出保真: 全局单例时间槽位, 与节点数解耦 (见 NextProcessTimes).
#define NEXT_PROCESS_TIMES_CAP 4096

// 019 扇出保真: 单节点最大对端数 (127 实测, 取 128).
#define MAX_FANOUT_PEERS_PER_NODE 128
// 036 (layer0): 单 NPU 扇出节点数上限 (实测 4 = 2 OUTCAST + 2 INCAST, 取 16).
#define MAX_FANOUT_NODES_PER_NPU 16
// 036 (layer0): 单 NPU 并发 INCAST 等齐上限 (实测 2, 取 16).
#define MAX_INCAST_WAITS_PER_NPU 16

namespace madsimple
{

    // ------------ Topology Ring-----------------
    enum class Direction
    {
        Clockwise,
        Anticlockwise
    };
    enum class Dimension
    {
        Local,
        Horizontal,
        Vertical,
        NA
    };

    // -----------------------------

    enum TaskState : int32_t
    {
        INIT = 0,
        START = 1,
        FINISH = 2
    };

    struct ProcessingCommTask
    {
        int64_t time_start_ns;
        int64_t time_end_ns;
        // means process times
        int64_t time_finish_ns;
        TaskState state;
        int32_t node_id;

        // Default constructor
        ProcessingCommTask()
            : time_start_ns(0),      // Initialize to 0
              time_end_ns(0),      // Initialize to 0
              time_finish_ns(0),      // Initialize to 0
              state(TaskState::INIT), // Initialize to INIT
              node_id(-1)
        { // Initialize to -1
        }

        void init()
        {
            time_start_ns = 0;
            time_end_ns = 0;
            time_finish_ns = 0;
            state = TaskState::INIT;
            node_id = -1;
        }
    };


    struct SysFlow
    {
        uint32_t id;
        uint64_t comm_size;
        uint64_t comm_src;
        uint64_t comm_dst;
        uint64_t comm_para;
        uint32_t durationMicros;

        // Flow execution order ID
        int exec_index;
        // Whether execution is completed
        TaskState state;
        // Send completion or receive completion
        bool is_send;

        SysFlow()
            : id(INVALID_FLOW_ID), // Initialize to -1
              comm_size(0),
              comm_src(0),
              comm_dst(0),
              comm_para(0),
              durationMicros(0),
              exec_index(0),
              state(TaskState::INIT),
              is_send(true)
        { // Initialize to true
        }

        void init()
        {
            id = INVALID_FLOW_ID;
            comm_size = 0;
            comm_src = 0;
            comm_dst = 0;
            comm_para = 0;
            durationMicros = 0;
            exec_index = 0;
            state = TaskState::INIT;
            is_send = true;
        }
    };

    struct NpuID
    {
        uint32_t value;
        bool test_dim;
        bool test_collective_comm_ring;

        SysFlow flows_exec[MAX_FLOW_NUM_PER_COMM_NODE];
        SysFlow flows_finish[MAX_FLOW_NUM_PER_COMM_NODE];
        SysFlow recv_node_flows_finish[MAX_FLOW_NUM_PER_COMM_NODE];
        ProcessingCommTask result[MAX_COMM_TASK_PER_NPU];

        void result_init()
        {
            for (int i = 0; i < MAX_COMM_TASK_PER_NPU; i++)
            {
                result[i].init();
            }
        }

        void flows_exec_init()
        {
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; i++)
            {
                flows_exec[i].init();
            }
        }

        void flows_finish_init()
        {
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; i++)
            {
                flows_finish[i].init();
            }
        }

        void recv_node_flows_finish_init()
        {
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; i++)
            {
                recv_node_flows_finish[i].init();
            }
        }


        NpuID()
            : value(0),
              test_dim(false),
              test_collective_comm_ring(false)
        {
        }
    };

    struct NodeID
    {
        uint32_t value;
    };

    enum class ChakraNodeType : int32_t
    {
        None = 0,
        COMP_NODE = 1,
        COMM_SEND_NODE = 2,
        COMM_RECV_NODE = 3,
        COMM_COLL_NODE = 4,
        // 019 扇出保真: 1 节点扇出 127 流 (outcast) / 等齐 127 流 (incast).
        OUTCAST_NODE = 5,
        INCAST_NODE = 6
    };

    enum CollectiveCommType : uint64_t
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
        Ring = 0,
        Tree = 1
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
        CommImplementationType reduce_implementation;
        CommImplementationType all_gather_implementation;
        CommImplementationType gather_implementation;
        CommImplementationType scatter_implementation;
        CommImplementationType broadcast_implementation;
        CommImplementationType all_to_all_implementation;
        CommImplementationType reduce_scatter_implementation;
        CommImplementationType reduce_scatter_block_implementation;
        CommImplementationType barrier_implementation;
    };

    struct SysConfig : public madrona::Archetype<CommModel>
    {
    };




    struct TaskFlows
    {
        bool is_exec;
        int current_index;
        SysFlow flows[MAX_FLOW_NUM_PER_COMM_NODE];

        TaskFlows()
        {
            is_exec = false;
            current_index = -1;
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; ++i)
            {
                flows[i] = SysFlow(); // Initialize each task
            }
        }

        void setCurrentIndexFlowsFinish()
        {
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; ++i)
            {
                if (flows[i].id == INVALID_FLOW_ID || flows[i].state == TaskState::FINISH)
                    continue; // Skip uninitialized flows
                if ((int)flows[i].exec_index == current_index)
                {
                    flows[i].state = TaskState::FINISH;
                }
            }
        }

        int getNextExecFlows(SysFlow flows_out[])
        {
            int min_exec_index = -1;
            int count = 0;

            // First pass: find the smallest exec_index greater than current_index
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; ++i)
            {
                const SysFlow &flow = flows[i];
                if (flow.id == INVALID_FLOW_ID || flow.state == TaskState::FINISH)
                    continue;
                if ((int)flow.exec_index > current_index)
                {
                    if (min_exec_index == -1 || flow.exec_index < min_exec_index)
                    {
                        min_exec_index = flow.exec_index;
                    }
                }
            }

            // If no exec_index greater than current_index is found, return 0
            if (min_exec_index == -1)
            {
                return 0;
            }
            else
            {
                current_index = min_exec_index;
            }

            // Second pass: collect all flows with exec_index equal to min_exec_index
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; ++i)
            {
                const SysFlow &flow = flows[i];
                if (flow.id == INVALID_FLOW_ID)
                    continue;
                if (flow.exec_index == min_exec_index)
                {
                    flows_out[count++] = flow;
                }
            }
            return count;
        }

        // Check if there exists a flow with exec_index equal to current_index
        bool hasFlowWithCurrentIndex() const
        {
            for (int i = 0; i < MAX_FLOW_NUM_PER_COMM_NODE; ++i)
            {
                const SysFlow &flow = flows[i];
                if (flow.id == INVALID_FLOW_ID || flow.state == TaskState::FINISH)
                    continue; // Skip uninitialized flows
                if (flow.exec_index == current_index && flow.state == TaskState::START)
                {
                    return true;
                }
            }
            return false;
        }

        void updateFlows(const SysFlow flows_finish[], int flows_finish_size)
        {
            for (int i = 0; i < flows_finish_size; ++i)
            {
                const SysFlow &finishedFlow = flows_finish[i];
                for (int j = 0; j < MAX_FLOW_NUM_PER_COMM_NODE; ++j)
                {
                    if (flows[j].id == finishedFlow.id)
                    { // Find element with same ID
                        flows[j].state = TaskState::FINISH;
                        flows[j].is_send = finishedFlow.is_send;               // Assign is_send
                        flows[j].durationMicros = finishedFlow.durationMicros; // Assign durationMicros
                        break;                                                 // Break inner loop after finding match
                    }
                }
            }
        }

        // Check if all tasks are completed
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
        ChakraNodeType type;
        uint32_t id;
        uint32_t data_deps[10];
        uint64_t comm_para;
        uint64_t comm_size;
        uint64_t comm_src;
        uint64_t comm_dst;
        CollectiveCommType comm_type;
        bool involved_dim_1;
        bool involved_dim_2;
        bool involved_dim_3;
        uint32_t durationMicros;

        // Copy constructor
        ChakraNode(const ChakraNode &other)
        {
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
            comm_type = other.comm_type;
            involved_dim_1 = other.involved_dim_1;
            involved_dim_2 = other.involved_dim_2;
            involved_dim_3 = other.involved_dim_3;
            durationMicros = other.durationMicros;
        }

        // Default constructor
        ChakraNode() = default;

        // Custom assignment operator
        ChakraNode &operator=(const ChakraNode &other)
        {
            if (this == &other)
                return *this; // Prevent self-assignment

            // Copy each member variable
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
            comm_type = other.comm_type;
            involved_dim_1 = other.involved_dim_1;
            involved_dim_2 = other.involved_dim_2;
            involved_dim_3 = other.involved_dim_3;
            durationMicros = other.durationMicros;

            return *this;
        }

        // Convert name from uint32_t array to readable string
        void getNameAsString(char output[64]) const
        {
            int output_len = 0;
            for (int i = 0; i < 20 && output_len < 63; ++i)
            {
                if (name[i] == 0 || name[i] == 2147483646 || name[i] > 127)
                    break;
                output[output_len] = static_cast<char>(name[i]);
                output_len++;
            }
            output[output_len] = '\0';
        }

        // Print name in readable format
        void printName() const
        {
            printf("name: ");
            for (int i = 0; i < 20; ++i)
            {
                if (name[i] == 0 || name[i] == 2147483646 || name[i] > 127)
                    break;
                printf("%c", static_cast<char>(name[i]));
            }
            printf("\n");
        }

        // Get name length
        int getNameLength() const
        {
            int length = 0;
            for (int i = 0; i < 20; ++i)
            {
                if (name[i] == 0 || name[i] == 2147483646 || name[i] > 127)
                    break;
                length++;
            }
            return length;
        }

        // Set name from character array
        void setNameFromChars(const char input[64])
        {
            int i = 0;
            while (i < 20 && input[i] != '\0')
            {
                name[i] = static_cast<uint32_t>(input[i]);
                i++;
            }
            while (i < 20)
            {
                name[i] = 2147483646; // Use same placeholder as Python script
                i++;
            }
        }

        void print()
        {
            printf("id: %d\n", id);
            
            // Print name using the new GPU device function
            printName();
            
            printf("data_deps: ");
            for (int i = 0; i < 10; ++i)
            {
                printf("%d ", data_deps[i]);
            }
            printf("\n");
            printf("type: %d\n", static_cast<int32_t>(type));
            printf("comm_para: %lu\n", comm_para);
            printf("comm_size: %lu\n", comm_size);
            printf("comm_src: %lu\n", comm_src);
            printf("comm_dst: %lu\n", comm_dst);
            printf("comm_type: %lu\n", comm_type);
            printf("involved_dim_1: %d\n", involved_dim_1);
            printf("involved_dim_2: %d\n", involved_dim_2);
            printf("involved_dim_3: %d\n", involved_dim_3);
            printf("durationMicros: %d\n", durationMicros);
        }
    };

    struct ChakraNodesData
    {
        uint32_t data[CHAKRA_DATA_ROWS][CHAKRA_NODES_DATA_LENGTH];
    };

    // Compatibility layout consumed by the migrated system layer. Python
    // builds this through a typed SystemConfig wrapper.
    struct ProcessParams
    {
        int32_t params[1000] {};
    };

    struct SystemStatus
    {
        int32_t initialized = 0;
        int32_t finished = 0;
        int32_t failed = 0;
        // Error semantics (latched with failed=1, first writer wins is not
        // guaranteed -- codes are distinct per failure family):
        //   1 = invalid system-layer init, 2/3 = bad comm src/dst,
        //   4 = NPU flow inbox/finished-list overflow,
        //   5 = NPU pair-state overflow,
        //   6 = delayed-event drop / registration loss,
        //   7 = C1 comm task entity pool exhausted (per-NPU preallocated
        //       ProcessComm_E/ProcessRecvComm_E pool empty; raise
        //       MAX_COMM_ENTITIES_PER_NPU),
        //   100+ = InferenceError (invalid config, capacity, queue
        //       overflow, ...).
        int32_t error_code = 0;
        int32_t finished_npus = 0;
        int32_t total_npus = 0;
        uint64_t next_system_event_ns = 0;
    };

    struct SysInputArch : public madrona::Archetype<
        ChakraNodesData,
        ProcessParams,
        InferenceConfigData,
        InferenceRequestData,
        WorkloadParamsTableData>
    {};

    struct ChakraNodes
    {
        ChakraNode nodes[MAX_CHAKRA_NODES_PER_NPU];
    };

    // 036 只读共享: 4P+4D 实例复用 256 份只读模板 (128 P + 128 D),
    // 单例存放, 省去每 NPU 独立 ChakraTemplateNodes 组件 (约 115MB/世界).
    struct SharedChakraTemplates
    {
        ChakraNode nodes[CHAKRA_DATA_ROWS][MAX_CHAKRA_NODES_PER_NPU];
    };

    struct ChakraNodesForNoDP
    {
        ChakraNode current_exec_nodes[CURRENT_EXEC_NODES_MAX];
    };

    // B1 incremental "ready node" bookkeeping, one component per NPU
    // (replaces the per-frame full O(193) filterNoDependencyNodes scan).
    // Invariant maintained by buildChakraReadyState / removeNode:
    //  - valid_count == number of slots whose node type != None
    //    (the legacy is_none_node == (valid_count == 0));
    //  - remaining_deps[i] == number of non-INVALID dependency entries on
    //    slot i;
    //  - ready[0..ready_count) == the slots with zero remaining
    //    dependencies, kept in ASCENDING slot order -- the exact emission
    //    order the legacy full scan produced (comp-device first-come-
    //    first-served and per-NPU flow-id numbering both consume this
    //    order, so it must not drift).
    struct ChakraReadyState
    {
        int32_t ready[MAX_CHAKRA_NODES_PER_NPU] {};
        int32_t ready_count = 0;
        int32_t remaining_deps[MAX_CHAKRA_NODES_PER_NPU] {};
        int32_t valid_count = 0;
    };


    struct HardwareResource
    {
        bool comp_ocupy;
        // bool comm_ocupy;
        bool one_task_finish;

        HardwareResource()
            : comp_ocupy(false), // Initialize to false
              one_task_finish(true)
        { // Initialize to true
        }
    };

    struct NextProcessTimes
    {
        // 019: 全局单例, 1024 NPU 共享. 稳态每 NPU 约 1 个在途 comp 事件;
        // 与 MAX_CHAKRA_NODES_PER_NPU 解耦, 取 4096 (约 4x 余量) 防静默丢事件.
        uint64_t times_abs[NEXT_PROCESS_TIMES_CAP];
    };

    struct NextProcessTimeE : public madrona::Archetype<NextProcessTimes>
    {
    };

    struct OneNPUFinishedFlag
    {
        bool is_finished;
        OneNPUFinishedFlag()
            : is_finished(false)
        { // Initialize to false
        }
    };

    //check finish
    struct CheckNpuFinishFlag
    {
        int32_t counter;
        CheckNpuFinishFlag()
            : counter(0)
        { // Initialize to 0
        }
    };

    struct CheckNpuFinishE : public madrona::Archetype<CheckNpuFinishFlag>
    {
    };

    struct ProcessingCompTask
    {
        // means process times
        int64_t time_finish_ns;
        TaskState state;
        int32_t node_id;

        // Default constructor
        ProcessingCompTask()
            : time_finish_ns(0),      // Initialize to 0
              state(TaskState::INIT), // Initialize to INIT
              node_id(-1)
        { // Initialize to -1 (invalid node)
        }
    };

   

    struct ProcessingCommTasks
    {
        ProcessingCommTask tasks[MAX_COMM_TASK_PER_NPU];

        int64_t flow_id;

        // B1 per-NPU duplicate-check acceleration: node_task_count[id]
        // counts this NPU's task slots whose state != INIT and whose
        // node_id == id, so containsNodeId is O(1) instead of a 512-slot
        // linear scan. Maintained by startTask / dequeueTasksByTime /
        // getNextFreeTask's overflow fallback; reset by the ProcessingCommTasks
        // default assignment in sys_init / resetNpuForReplay. Ids outside
        // [0, MAX_COMM_TASK_PER_NPU) fall back to the legacy linear scan.
        int32_t node_task_count[MAX_COMM_TASK_PER_NPU] {};

        // Default constructor
        ProcessingCommTasks()
        {
            for (int i = 0; i < MAX_COMM_TASK_PER_NPU; ++i)
            {
                tasks[i] = ProcessingCommTask(); // Initialize each task
            }
        }

        // Get total flow count of all tasks
        int getFlowId()
        {
            flow_id += 1;
            return flow_id;
        }

        // Check if a node with specified ID exists
        bool containsNodeId(int32_t node_id) const
        {
            // B1: O(1) path via the per-node live-task counts.
            if (node_id >= 0 && node_id < MAX_COMM_TASK_PER_NPU) {
                return node_task_count[node_id] > 0;
            }
            // Fallback (ids outside the count-array range): exact legacy
            // linear scan.
            for (int i = 0; i < MAX_COMM_TASK_PER_NPU; ++i)
            {
                // Task state may be START or FINISH, not yet processed
                if (tasks[i].state != TaskState::INIT && tasks[i].node_id == node_id)
                {
                    return true; // Found matching task
                }
            }
            return false; // No matching task found
        }

        // B1 bookkeeping: retire/acquire the per-node live-task count.
        void retireNodeCount(int32_t node_id)
        {
            if (node_id >= 0 && node_id < MAX_COMM_TASK_PER_NPU &&
                node_task_count[node_id] > 0) {
                node_task_count[node_id] -= 1;
            }
        }

        void acquireNodeCount(int32_t node_id)
        {
            if (node_id >= 0 && node_id < MAX_COMM_TASK_PER_NPU) {
                node_task_count[node_id] += 1;
            }
        }

        // B1: register a task slot as started (INIT -> START) with its node
        // id, keeping the per-node count exact. Handles the
        // getNextFreeTask() overflow fallback where an occupied slot is
        // reused: the previous occupant's count is retired first so the
        // counts keep matching the linear-scan semantics exactly.
        void startTask(ProcessingCommTask &task, int32_t node_id)
        {
            const int32_t idx = static_cast<int32_t>(&task - tasks);
            if (idx < 0 || idx >= MAX_COMM_TASK_PER_NPU) {
                return;
            }
            if (tasks[idx].state != TaskState::INIT) {
                retireNodeCount(tasks[idx].node_id);
            }
            task.node_id = node_id;
            task.state = TaskState::START;
            acquireNodeCount(node_id);
        }

        void setFinish(int32_t node_id, int64_t time_finish_ns,uint32_t npu_id)
        {
            for (int i = 0; i < MAX_COMM_TASK_PER_NPU; ++i)
            {
                if (tasks[i].state == TaskState::START && tasks[i].node_id == node_id)
                {
                    #if SYS_LOG
                    if (SYS_LOG_TARGET_NODE == npu_id) {
                    printf("node id %d -> setFinish\n", node_id);
                    }
                    #endif
                    #if SIMPLE_LOG_MODE
                    if (SYS_LOG_TARGET_NODE == npu_id) {
                    printf("node id %d -> setFinish\n", node_id);
                    }
                    #endif
                    tasks[i].state = TaskState::FINISH;
                    tasks[i].time_end_ns = time_finish_ns;
                    tasks[i].time_finish_ns = time_finish_ns-tasks[i].time_start_ns;
                    break;
                }
            }
        }

        // Method to add a task
        bool addTask(const ProcessingCommTask &new_task)
        {
            for (int i = 0; i < MAX_COMM_TASK_PER_NPU; ++i)
            {
                if (tasks[i].state == TaskState::INIT)
                {                                      // Find first position with node_id == -1
                    tasks[i] = new_task;               // Place new task
                    tasks[i].state = TaskState::START; // Mark task as valid
                    acquireNodeCount(new_task.node_id); // B1: keep count exact
                    return true;                       // Add successful
                }
            }
            return false; // No free position, add failed
        }

        ProcessingCommTask& getNextFreeTask()
        {
            for (int i = 0; i < MAX_COMM_TASK_PER_NPU; ++i)
            {
                if (tasks[i].state == TaskState::INIT)
                {
                    return tasks[i];
                }
            }
            return tasks[0];
        }

        // Check if at least one node with is_none == false exists
        bool has_task() const
        {
            for (int i = 0; i < MAX_COMM_TASK_PER_NPU; ++i)
            {
                if (tasks[i].state == TaskState::START || tasks[i].state == TaskState::FINISH)
                {
                    // If found a node with is_none == false
                    return true;
                }
            }
            return false; // If all nodes have is_none == true
        }

        // Find all tasks with time_finish_ns <= t and set their is_none to true
        int dequeueTasksByTime(int64_t t, ProcessingCommTask result[], int max_result_size)
        {
            int count = 0;
            for (int i = 0; i < MAX_COMM_TASK_PER_NPU && count < max_result_size; ++i)
            {
                if (tasks[i].state == TaskState::FINISH && tasks[i].time_end_ns <= t)
                {
                    result[count++] = tasks[i];       // Place task in result array
                    retireNodeCount(tasks[i].node_id); // B1: keep per-node count exact
                    tasks[i].state = TaskState::INIT; // Mark task as invalid (dequeue)
                }
            }
            return count; // Return number of tasks found
        }
    };


    struct RecvNodeFlag
    {
        uint64_t comm_src;
        uint64_t comm_dst;
        uint64_t flow_id;
    };

    // 019 扇出保真: 主表后 side 区解码出的 (peers, paras) 清单.
    // COMP 节点不占此空间 (与 ChakraNode 分离, 避免撑大逐节点结构);
    // rank/para 均 < 2^32, 用 uint32_t (单 NPU 约 256x1KB = 256KB).
    struct ChakraFanoutEntry
    {
        uint32_t node_id = 0;
        uint32_t count = 0;
        uint32_t peers[MAX_FANOUT_PEERS_PER_NODE] {};
        uint32_t paras[MAX_FANOUT_PEERS_PER_NODE] {};
    };

    struct ChakraFanouts
    {
        uint32_t num_entries = 0;
        ChakraFanoutEntry entries[MAX_FANOUT_NODES_PER_NPU] {};
    };

    // 019: INCAST 等齐进度 (逐帧轮询 claimRecvDone, 127 位用 2x u64 记录).
    // 不占 ProcessRecvComm_E 实体池 (122 并发 INCAST x 127 实体会打爆
    // 256 池), 只占一个 ProcessingCommTasks 槽位 + 本表一条目.
    struct IncastWaitEntry
    {
        uint32_t node_id = 0;
        uint32_t total = 0;
        uint32_t done = 0;
        uint64_t claimed[2] {};
        int32_t fanout_idx = -1;
    };

    struct IncastWaitList
    {
        uint32_t count = 0;
        uint32_t overflow_count = 0;
        IncastWaitEntry waits[MAX_INCAST_WAITS_PER_NPU] {};
    };

    struct NpuNode : public madrona::Archetype<
                         NpuID,
                         ChakraNodes,
                         InferenceNpuExecution,
                         HardwareResource,
                         ProcessingCompTask,
                         ProcessingCommTasks,
                         OneNPUFinishedFlag,
                         ChakraNodesForNoDP,
                         ChakraReadyState,
                         ChakraFanouts,
                         IncastWaitList,
                         NpuFlowInbox,
                         NpuFlowPool,
                         NpuFlowActiveList,
                         NpuFlowFinishedList,
                         NpuFlowPairState,
                          NpuKvCollectDraft,
                          NpuFlowScheduleDraft,
                          NpuFlowCreateDraft,
                          NpuCommEntityPool,
                         NpuCommRetireDraft>
    {
    };



    // ID &id, CollectiveCommType &collective_comm_type, CommParams &comm_params,TaskFlows &taskFlows
    struct ProcessComm_E : public madrona::Archetype<
                               NpuID, NodeID,
                               TaskFlows,
                               PoolSlotActive>
    {
    };

    // Ring Parameters for collective communication
    struct RingParams
    {
        uint32_t ring_dim0;
        uint32_t ring_dim1;
        uint32_t ring_dim2;
        uint32_t chunks_num;
        uint32_t npu_num;
        // Default constructor
        RingParams() : ring_dim0(0), ring_dim1(0), ring_dim2(0), chunks_num(0), npu_num(0) {}
    };

    // Entity archetype to store ring configuration parameters
    struct RingConfigEntity : public madrona::Archetype<RingParams>
    {
    };



    struct ProcessRecvComm_E : public madrona::Archetype<
    NpuID, NodeID,
    RecvNodeFlag,
    PoolSlotActive>
    {
    };
} // namespace madsimple
