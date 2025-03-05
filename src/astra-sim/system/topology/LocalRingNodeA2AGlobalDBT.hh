/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/topology/ComplexLogicalTopology.hh"
#include "astra-sim/system/topology/DoubleBinaryTreeTopology.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {

// 局部环形节点全对全全局双二叉树拓扑类
class LocalRingNodeA2AGlobalDBT : public ComplexLogicalTopology {
public:
    CUDA_HOST_DEVICE LocalRingNodeA2AGlobalDBT(int id, int local_dim, int node_dim, int total_tree_nodes, int start, int stride);
    CUDA_HOST_DEVICE ~LocalRingNodeA2AGlobalDBT();
    
    CUDA_HOST_DEVICE int get_num_of_nodes_in_dimension(int dimension) override;
    CUDA_HOST_DEVICE BasicLogicalTopology* get_basic_topology_at_dimension(int dimension, ComType type) override;
    CUDA_HOST_DEVICE int get_num_of_dimensions() override;

    DoubleBinaryTreeTopology* global_dimension_all_reduce;
    RingTopology* global_dimension_other;
    RingTopology* local_dimension;
    RingTopology* node_dimension;
};

}  // namespace AstraSim
