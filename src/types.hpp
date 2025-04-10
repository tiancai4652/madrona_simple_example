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

struct ID {
    uint32_t value;
};

enum class NodeType : int32_t {
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

struct ChakraNodes {
    uint32_t name[20],
    NodeType type,
    uint32_t id,
    uint32_t data_deps[10],
    

};

struct NpuNode : public madrona::Archetype<
    ID,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep,
    ChakraNodesData
> {};

// ------------------------------------------

}
