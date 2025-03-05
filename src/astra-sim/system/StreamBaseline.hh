/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/CollectivePhase.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "sys_layer/containers/FixedList.hpp"

namespace AstraSim {

class Sys;

/**
 * 基准流类
 */
class StreamBaseline : public BaseStream {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    StreamBaseline(Sys* owner, 
                  DataSet* dataset, 
                  int stream_id, 
                  custom::FixedList<CollectivePhase, 32> phases_to_go, 
                  int priority);

    /**
     * 初始化流
     */
    CUDA_HOST_DEVICE
    void init() override;

    /**
     * 调用事件处理
     */
    CUDA_HOST_DEVICE
    void call(EventType event, CallData* data) override;

    /**
     * 消费接收到的数据包
     */
    CUDA_HOST_DEVICE
    void consume(RecvPacketEventHandlerData* message) override;
};

}  // namespace AstraSim
