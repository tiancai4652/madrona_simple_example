/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/collective/Algorithm.hh"
#include "astra-sim/system/topology/BinaryTree.hh"

namespace AstraSim {

/**
 * 双二叉树全规约算法实现
 */
class DoubleBinaryTreeAllReduce : public Algorithm {
  public:
    /// 算法状态枚举
    enum class State {
        Begin = 0,                ///< 开始状态
        WaitingForTwoChildData,   ///< 等待两个子节点数据
        WaitingForOneChildData,   ///< 等待一个子节点数据
        SendingDataToParent,      ///< 向父节点发送数据
        WaitingDataFromParent,    ///< 等待父节点数据
        SendingDataToChilds,      ///< 向子节点发送数据
        End                       ///< 结束状态
    };

    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    DoubleBinaryTreeAllReduce(int id, BinaryTree* tree, uint64_t data_size);

    /**
     * 运行算法
     */
    CUDA_HOST_DEVICE
    void run(EventType event, CallData* data) override;

    BinaryTree::Type type;    ///< 节点类型
    State state;              ///< 当前状态
    int parent;              ///< 父节点ID
    int left_child;          ///< 左子节点ID
    int reductions;          ///< 规约次数
    int right_child;         ///< 右子节点ID
};

}  // namespace AstraSim
