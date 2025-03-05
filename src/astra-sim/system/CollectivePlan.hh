//
// Created by Saeed Rashidi on 7/7/22.
//

#pragma once

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/topology/LogicalTopology.hh"
#include "sys_layer/containers/FixedVector.hpp"

namespace AstraSim {

/**
 * 集体通信计划类
 */
class CollectivePlan {
public:
    CUDA_HOST_DEVICE CollectivePlan(
        LogicalTopology* topology,
        custom::FixedVector<CollectiveImpl*, 32> implementation_per_dimension,
        custom::FixedVector<bool, 32> dimensions_involved,
        bool should_be_removed);
    CUDA_HOST_DEVICE ~CollectivePlan();

    LogicalTopology* topology;                                    ///< 逻辑拓扑
    custom::FixedVector<CollectiveImpl*, 32> implementation_per_dimension;  ///< 每个维度的实现
    custom::FixedVector<bool, 32> dimensions_involved;           ///< 参与的维度
    bool should_be_removed;                                      ///< 是否应该被移除
};

}  // namespace AstraSim
