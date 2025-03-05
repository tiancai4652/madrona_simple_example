/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <assert.h>
#include "sys_layer/containers/FixedVector.hpp"
#include "sys_layer/containers/FixedMap.hpp"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class CollectivePlan;

/**
 * 通信器组类，用于管理集体通信
 */
class CommunicatorGroup {
public:
    CUDA_HOST_DEVICE CommunicatorGroup(int id, custom::FixedVector<int, 32> involved_NPUs, Sys* generator);
    CUDA_HOST_DEVICE CollectivePlan* get_collective_plan(ComType comm_type);
    CUDA_HOST_DEVICE void set_id(int id);
    CUDA_HOST_DEVICE ~CommunicatorGroup();

    custom::FixedVector<int, 32> involved_NPUs;  ///< 参与的NPU列表
    int num_streams;                             ///< 流的数量

private:
    int id;                                      ///< 通信器组ID
    Sys* generator;                              ///< 系统生成器
    custom::FixedMap<ComType, CollectivePlan*, 32> comm_plans;  ///< 通信计划映射
};

}  // namespace AstraSim
