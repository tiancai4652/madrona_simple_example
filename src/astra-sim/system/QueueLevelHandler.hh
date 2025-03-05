/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/topology/RingTopology.hh"
#include "sys_layer/containers/FixedVector.hpp"

namespace AstraSim {

/**
 * 队列层级处理器类
 */
class QueueLevelHandler {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    QueueLevelHandler(int level, int start, int end, AstraNetworkAPI::BackendType backend);

    /**
     * 获取下一个队列ID
     */
    CUDA_HOST_DEVICE
    std::pair<int, RingTopology::Direction> get_next_queue_id();

    /**
     * 获取第一个队列ID
     */
    CUDA_HOST_DEVICE
    std::pair<int, RingTopology::Direction> get_next_queue_id_first();

    /**
     * 获取最后一个队列ID
     */
    CUDA_HOST_DEVICE
    std::pair<int, RingTopology::Direction> get_next_queue_id_last();

    custom::FixedVector<int, 32> queues;  ///< 队列列表
    int allocator;                       ///< 当前分配器位置
    int first_allocator;                 ///< 第一个分配器位置
    int last_allocator;                  ///< 最后一个分配器位置
    int level;                          ///< 层级
    AstraNetworkAPI::BackendType backend; ///< 后端类型
};

}  // namespace AstraSim
