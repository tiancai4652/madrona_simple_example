/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

class WorkloadLayerHandlerData;

/**
 * 接收数据包事件处理器数据类
 */
class RecvPacketEventHandlerData : public BasicEventHandlerData {
  public:
    /**
     * 默认构造函数
     */
    CUDA_HOST_DEVICE
    RecvPacketEventHandlerData();

    /**
     * 带参数构造函数
     */
    CUDA_HOST_DEVICE
    RecvPacketEventHandlerData(BaseStream* owner, int sys_id, EventType event, int vnet, int stream_id);

    Workload* workload;                  ///< 工作负载指针
    WorkloadLayerHandlerData* wlhd;      ///< 工作负载层处理器数据
    BaseStream* owner;                   ///< 所属流
    int vnet;                           ///< 虚拟网络ID
    int stream_id;                      ///< 流ID
    bool message_end;                   ///< 消息结束标志
    Tick ready_time;                    ///< 就绪时间
};

}  // namespace AstraSim
