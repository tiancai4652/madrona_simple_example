/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/topology/ComplexLogicalTopology.hh"
#include "astra-sim/system/topology/LocalRingGlobalBinaryTree.hh"

namespace AstraSim {

// 双二叉树拓扑类
class DoubleBinaryTreeTopology : public ComplexLogicalTopology {
public:
    CUDA_HOST_DEVICE DoubleBinaryTreeTopology(int id, int total_tree_nodes, int start, int stride);
    CUDA_HOST_DEVICE ~DoubleBinaryTreeTopology();
    
    CUDA_HOST_DEVICE LogicalTopology* get_topology() override;
    CUDA_HOST_DEVICE BasicLogicalTopology* get_basic_topology_at_dimension(int dimension, ComType type) override;
    CUDA_HOST_DEVICE int get_num_of_dimensions() override;
    CUDA_HOST_DEVICE int get_num_of_nodes_in_dimension(int dimension) override;

    int counter;
    BinaryTree* DBMAX;
    BinaryTree* DBMIN;
};

}  // namespace AstraSim
