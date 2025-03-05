/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

/**
 * 自定义数据包类
 */
class MyPacket : public Callable {
  public:
    /**
     * 构造函数 - 基本参数
     */
    CUDA_HOST_DEVICE
    MyPacket(int preferred_vnet, int preferred_src, int preferred_dest);

    /**
     * 构造函数 - 带消息大小
     */
    CUDA_HOST_DEVICE
    MyPacket(uint64_t msg_size, int preferred_vnet, int preferred_src, int preferred_dest);

    /**
     * 设置通知者
     */
    CUDA_HOST_DEVICE
    void set_notifier(Callable* c);

    /**
     * 调用处理
     */
    CUDA_HOST_DEVICE
    void call(EventType event, CallData* data) override;

    int cycles_needed;                ///< 需要的周期数
    int fm_id;                       ///< 特征图ID
    int stream_id;                   ///< 流ID
    Callable* notifier;              ///< 通知者
    Callable* sender;                ///< 发送者
    int preferred_vnet;              ///< 首选虚拟网络
    int preferred_dest;              ///< 首选目标
    int preferred_src;               ///< 首选源
    uint64_t msg_size;               ///< 消息大小
    Tick ready_time;                 ///< 就绪时间
};

}  // namespace AstraSim
