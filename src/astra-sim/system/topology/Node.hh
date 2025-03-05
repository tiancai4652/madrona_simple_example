/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/topology/ComputeNode.hh"

namespace AstraSim {

// 二叉树节点类
class Node : public ComputeNode {
public:
    CUDA_HOST_DEVICE Node(int id, Node* parent, Node* left_child, Node* right_child);

    int id;
    Node* parent;
    Node* left_child;
    Node* right_child;
};

}  // namespace AstraSim
