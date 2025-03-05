/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <map>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/topology/BasicLogicalTopology.hh"
#include "astra-sim/system/topology/Node.hh"

namespace AstraSim {

// 二叉树拓扑类
class BinaryTree : public BasicLogicalTopology {
public:
    enum class TreeType { 
        RootMax, 
        RootMin 
    };
    
    enum class Type { 
        Leaf, 
        Root, 
        Intermediate 
    };

    CUDA_HOST_DEVICE BinaryTree(int id, TreeType tree_type, int total_tree_nodes, int start, int stride);
    CUDA_HOST_DEVICE virtual ~BinaryTree();

    CUDA_HOST_DEVICE int get_num_of_nodes_in_dimension(int dimension) override {
        return total_tree_nodes;
    }

    CUDA_HOST_DEVICE Node* initialize_tree(int depth, Node* parent);
    CUDA_HOST_DEVICE void build_tree(Node* node);
    CUDA_HOST_DEVICE int get_parent_id(int id);
    CUDA_HOST_DEVICE int get_left_child_id(int id);
    CUDA_HOST_DEVICE int get_right_child_id(int id);
    CUDA_HOST_DEVICE Type get_node_type(int id);
    CUDA_HOST_DEVICE void print(Node* node);

    int total_tree_nodes;
    int start;
    TreeType tree_type;
    int stride;
    Node* tree;
    std::map<int, Node*> node_list;
};

} // namespace AstraSim
