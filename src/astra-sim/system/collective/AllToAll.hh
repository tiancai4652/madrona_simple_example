/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/collective/Ring.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {

/**
 * 全对全集体通信算法实现
 */
class AllToAll : public Ring {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    AllToAll(ComType type,
             int window,
             int id,
             RingTopology* allToAllTopology,
             uint64_t data_size,
             RingTopology::Direction direction,
             InjectionPolicy injection_policy);

    /**
     * 运行算法
     */
    CUDA_HOST_DEVICE
    void run(EventType event, CallData* data) override;

    /**
     * 处理最大计数
     */
    CUDA_HOST_DEVICE
    void process_max_count() override;

    /**
     * 获取非零延迟包的数量
     */
    CUDA_HOST_DEVICE
    int get_non_zero_latency_packets() override;

    int middle_point;  ///< 中间点
};

}  // namespace AstraSim
