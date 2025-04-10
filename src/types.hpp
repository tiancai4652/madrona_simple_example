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
    uint32_t data[2][10000000];
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

// ----------------------------------------

#define INTS_PER_NODE  44  // 每个 ChakraNode 占用 44 字节，等于 11 个 int
#define MAX_CHAKRA_NODES 9*9999
#define chakra_nodes_data_length 10000000
// 定义无效依赖的值
#define INVALID_DEPENDENCY 4294967295

#define CURRENT_EXEC_NODES_MAX 10

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
    ChakraNode nodes[9*9999];
};


struct NpuNode : public madrona::Archetype<
    ID,
    ChakraNodes
> {};

// ------------------------------------------

}
