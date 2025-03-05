/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CallbackTrackerEntry.hh"
#include <cassert>
#include "sys_layer/containers/FixedOptional.hpp"

using namespace NetworkAnalytical;
using namespace AstraSimAnalytical;

CUDA_HOST_DEVICE
CallbackTrackerEntry::CallbackTrackerEntry() noexcept
    : send_event(make_nullopt<Event>()),
      recv_event(make_nullopt<Event>()),
      transmission_finished(false) {}

CUDA_HOST_DEVICE
void CallbackTrackerEntry::register_send_callback(const Callback callback, const CallbackArg arg) noexcept {
    assert(!send_event.has_value());

    // 注册发送回调
    const auto event = Event(callback, arg);
    send_event = make_optional(event);
}

CUDA_HOST_DEVICE
void CallbackTrackerEntry::register_recv_callback(const Callback callback, const CallbackArg arg) noexcept {
    assert(!recv_event.has_value());

    // 注册接收回调
    const auto event = Event(callback, arg);
    recv_event = make_optional(event);
}

CUDA_HOST_DEVICE
bool CallbackTrackerEntry::is_transmission_finished() const noexcept {
    return transmission_finished;
}

CUDA_HOST_DEVICE
void CallbackTrackerEntry::set_transmission_finished() noexcept {
    transmission_finished = true;
}

CUDA_HOST_DEVICE
bool CallbackTrackerEntry::both_callbacks_registered() const noexcept {
    // 检查两个回调是否都已注册
    return (send_event.has_value() && recv_event.has_value());
}

CUDA_HOST_DEVICE
void CallbackTrackerEntry::invoke_send_handler() noexcept {
    assert(send_event.has_value());

    // 触发发送事件
    send_event.value().invoke_event();
}

CUDA_HOST_DEVICE
void CallbackTrackerEntry::invoke_recv_handler() noexcept {
    assert(recv_event.has_value());

    // 触发接收事件
    recv_event.value().invoke_event();
}
