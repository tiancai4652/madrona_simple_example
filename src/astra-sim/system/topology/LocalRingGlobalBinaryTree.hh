/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/topology/BinaryTree.hh"
#include "astra-sim/system/topology/ComplexLogicalTopology.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {

// 局部环形全局二叉树拓扑类
class LocalRingGlobalBinaryTree : public ComplexLogicalTopology {
public:
    CUDA_HOST_DEVICE LocalRingGlobalBinaryTree(
        int id, int local_dim, BinaryTree::TreeType tree_type, int total_tree_nodes, int start, int stride);
    CUDA_HOST_DEVICE ~LocalRingGlobalBinaryTree();
    
    CUDA_HOST_DEVICE int get_num_of_nodes_in_dimension(int dimension) override;
    CUDA_HOST_DEVICE int get_num_of_dimensions() override;
    CUDA_HOST_DEVICE BasicLogicalTopology* get_basic_topology_at_dimension(int dimension, ComType type) override;

    RingTopology* local_dimension;
    RingTopology* global_dimension_other;
    BinaryTree* global_dimension_all_reduce;
};

}  // namespace AstraSim
