/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

class Workload;

/**
 * 工作负载层处理器数据类
 */
class WorkloadLayerHandlerData : public BasicEventHandlerData, public MetaData {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    WorkloadLayerHandlerData();

    int sys_id;                ///< 系统ID
    Workload* workload;        ///< 工作负载指针
    uint64_t node_id;          ///< 节点ID
};

}  // namespace AstraSim
