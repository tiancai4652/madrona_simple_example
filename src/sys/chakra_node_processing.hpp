#pragma once

#include "../sim.hpp"
#include "../types.hpp"

namespace madsimple::llm_system
{
    // Import madrona types into current namespace
    using madrona::ECSRegistry;

    // Chakra node parsing and processing functions
    void parseAttributeValue(AttributeDataType type, uint32_t data1, uint32_t data2, uint64_t &result);

    // MADRONA_NO_INLINE: large device functions (139L / 102L); keep them out
    // of callers' IR so NVRTC/ptxas per-function bodies stay small
    // (same rationale as sys_processChakraNodes in llm_system.hpp).
    MADRONA_NO_INLINE void parseChakraNodes(const ChakraNodesData &chakraNodesData, int row, ChakraNode parsedNodes[],
                          int &outNodeCount, int intArrayLength = CHAKRA_NODES_DATA_LENGTH, int maxNodes = MAX_CHAKRA_NODES_PER_NPU);

    int filterNoDependencyNodes(const ChakraNodes &chakraNodes, ChakraNode filteredNodes[],bool &is_none_node,
                                int maxNodes = CURRENT_EXEC_NODES_MAX);

    MADRONA_NO_INLINE int checkRecvNodeDependencies(const ChakraNodes &chakraNodes, uint32_t independentRecvNodes[], 
        int maxNodes = CURRENT_EXEC_NODES_MAX);

    void removeNode(ChakraNodes &chakraNodes, uint32_t nodeId);

    // B1 incremental ready-set maintenance (replaces the per-frame full
    // rescan; see ChakraReadyState in sys_types.hpp for the invariants).
    // Rebuild the whole ready state from the node array (used once at
    // parse / at resetNpuForReplay).
    void buildChakraReadyState(const ChakraNodes &chakraNodes,
                               ChakraReadyState &ready);
    // removeNode variant that also maintains the ready state. Semantically
    // identical to the legacy removeNode on ChakraNodes; the ready state is
    // pure bookkeeping consumed by sys_processChakraNodes.
    void removeNodeWithReady(ChakraNodes &chakraNodes, uint32_t nodeId,
                             ChakraReadyState &ready);
    // Fill filteredNodes[0..return) with the zero-dependency nodes in
    // ascending slot order (legacy filterNoDependencyNodes order), capped
    // at maxNodes. Sets is_none_node = (no valid nodes at all).
    int readyNodesFromState(const ChakraNodes &chakraNodes,
                            const ChakraReadyState &ready,
                            ChakraNode filteredNodes[], bool &is_none_node,
                            int maxNodes = CURRENT_EXEC_NODES_MAX);

    // Utility functions for node processing
    const char *CollectiveCommTypeToString(CollectiveCommType type);
    const char *DimensionToString(Dimension type);
}
