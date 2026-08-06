#pragma once

#include <madrona/sync.hpp>

#include "types.hpp"
#include "grid.hpp"

namespace madsimple {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct NodeDef {
    NodeId id = -1;
    NodeType type = NodeType::Host;
    Bw port_bw = 100.0;
};

struct LinkDef {
    NodeId src = -1;
    NodeId dst = -1;
    Time delay = 0.01;
    Bw bandwidth = 0.0;
};

struct FlowDef {
    FlowId id = -1;
    NodeId src_node = -1;
    NodeId dst_node = -1;
    Bytes size = 0.0;
    Time start_time = 0.0;
    int32_t priority = 0;
    uint64_t comm_para = 0;
};

// FlowCompletionRecord moved to types.hpp so it can be embedded in the
// FlowCompletionBuf component that backs the GPU-mode export path.

struct NetworkInit {
    const NodeDef *nodes;
    int32_t numNodes;
    const LinkDef *links;
    int32_t numLinks;
    const FlowDef *flows;
    int32_t numFlows;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    const GridState *grid;
    const NetworkInit *network;
};

}
