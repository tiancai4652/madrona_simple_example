/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MyPacket.hh"
#include "sys_layer/containers/FixedList.hpp"

namespace AstraSim {

class Sys;

/**
 * 数据包束类
 */
class PacketBundle : public Callable {
  public:
    /**
     * 构造函数 - 带锁定数据包
     */
    CUDA_HOST_DEVICE
    PacketBundle(Sys* sys,
                 BaseStream* stream,
                 custom::FixedList<MyPacket*, 32> locked_packets,
                 bool needs_processing,
                 bool send_back,
                 uint64_t size,
                 MemBus::Transmition transmition);

    /**
     * 构造函数 - 不带锁定数据包
     */
    CUDA_HOST_DEVICE
    PacketBundle(Sys* sys,
                 BaseStream* stream,
                 bool needs_processing,
                 bool send_back,
                 uint64_t size,
                 MemBus::Transmition transmition);

    /**
     * 发送到MA
     */
    CUDA_HOST_DEVICE
    void send_to_MA();

    /**
     * 发送到NPU
     */
    CUDA_HOST_DEVICE
    void send_to_NPU();

    /**
     * 调用处理
     */
    CUDA_HOST_DEVICE
    void call(EventType event, CallData* data) override;

    Sys* sys;                                      ///< 系统指针
    custom::FixedList<MyPacket*, 32> locked_packets; ///< 锁定的数据包列表
    bool needs_processing;                         ///< 是否需要处理
    bool send_back;                               ///< 是否需要发回
    uint64_t size;                                ///< 大小
    BaseStream* stream;                           ///< 流指针
    MemBus::Transmition transmition;              ///< 传输类型
    Tick delay;                                   ///< 延迟
    Tick creation_time;                           ///< 创建时间
};

}  // namespace AstraSim
