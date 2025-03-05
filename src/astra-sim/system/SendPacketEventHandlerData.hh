/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/BasicEventHandlerData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"

namespace AstraSim {

/**
 * 发送数据包事件处理器数据类
 */
class SendPacketEventHandlerData : public BasicEventHandlerData {
  public:
    int tag;                            ///< 消息标签
    Callable* callable;                 ///< 可调用对象
    WorkloadLayerHandlerData* wlhd;     ///< 工作负载层处理器数据

    /**
     * 默认构造函数
     */
    CUDA_HOST_DEVICE
    SendPacketEventHandlerData();

    /**
     * 带参数构造函数
     */
    CUDA_HOST_DEVICE
    SendPacketEventHandlerData(Callable* callable, int tag);
};

}  // namespace AstraSim
