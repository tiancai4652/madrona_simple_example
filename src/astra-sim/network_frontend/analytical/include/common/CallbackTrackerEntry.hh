/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <astra-network-analytical/common/Event.h>
#include <astra-network-analytical/common/Type.h>
#include "sys_layer/containers/FixedOptional.hpp"

using namespace NetworkAnalytical;

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSimAnalytical {

/**
 * CallbackTrackerEntry管理每个唯一块的sim_send()和sim_recv()回调
 */
class CallbackTrackerEntry {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    CallbackTrackerEntry() noexcept;

    /**
     * 注册sim_send()调用的回调
     */
    CUDA_HOST_DEVICE
    void register_send_callback(Callback callback, CallbackArg arg) noexcept;

    /**
     * 注册sim_recv()调用的回调
     */
    CUDA_HOST_DEVICE
    void register_recv_callback(Callback callback, CallbackArg arg) noexcept;

    /**
     * 检查块的传输是否已完成
     */
    CUDA_HOST_DEVICE
    [[nodiscard]] bool is_transmission_finished() const noexcept;

    /**
     * 将块的传输标记为已完成
     */
    CUDA_HOST_DEVICE
    void set_transmission_finished() noexcept;

    /**
     * 检查是否同时注册了sim_send()和sim_recv()回调
     */
    CUDA_HOST_DEVICE
    [[nodiscard]] bool both_callbacks_registered() const noexcept;

    /**
     * 调用sim_send()回调
     */
    CUDA_HOST_DEVICE
    void invoke_send_handler() noexcept;

    /**
     * 调用sim_recv()回调
     */
    CUDA_HOST_DEVICE
    void invoke_recv_handler() noexcept;

  private:
    /// sim_send()回调事件
    custom::FixedOptional<Event> send_event;

    /// sim_recv()回调事件
    custom::FixedOptional<Event> recv_event;

    /// 块的传输是否已完成
    bool transmission_finished;
};

}  // namespace AstraSimAnalytical
