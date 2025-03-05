/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/QueueLevelHandler.hh"
#include "astra-sim/system/topology/RingTopology.hh"
#include "sys_layer/containers/FixedVector.hpp"

namespace AstraSim {

/**
 * 队列层级管理类
 */
class QueueLevels {
  public:
    custom::FixedVector<QueueLevelHandler, 8> levels;  ///< 队列层级处理器列表

    /**
     * 获取指定层级的下一个队列
     */
    CUDA_HOST_DEVICE
    std::pair<int, RingTopology::Direction> get_next_queue_at_level(int level);

    /**
     * 获取指定层级的第一个队列
     */
    CUDA_HOST_DEVICE
    std::pair<int, RingTopology::Direction> get_next_queue_at_level_first(int level);

    /**
     * 获取指定层级的最后一个队列
     */
    CUDA_HOST_DEVICE
    std::pair<int, RingTopology::Direction> get_next_queue_at_level_last(int level);

    /**
     * 构造函数 - 均匀分布队列
     */
    CUDA_HOST_DEVICE
    QueueLevels(int levels, int queues_per_level, int offset, AstraNetworkAPI::BackendType backend);

    /**
     * 构造函数 - 自定义分布队列
     */
    CUDA_HOST_DEVICE
    QueueLevels(custom::FixedVector<int, 8> lv, int offset, AstraNetworkAPI::BackendType backend);
};

}  // namespace AstraSim
