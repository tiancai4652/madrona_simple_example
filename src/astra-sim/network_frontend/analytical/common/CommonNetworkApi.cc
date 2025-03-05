/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CommonNetworkApi.hh"
#include <cassert>
#include "sys_layer/containers/FixedVector.hpp"
#include "sys_layer/containers/FixedTuple.hpp"

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;

// 静态成员初始化
EventQueue* CommonNetworkApi::event_queue = nullptr;
ChunkIdGenerator<1024> CommonNetworkApi::chunk_id_generator;
CallbackTracker<1024> CommonNetworkApi::callback_tracker;
int CommonNetworkApi::dims_count = -1;
FixedVector<double, 8> CommonNetworkApi::bandwidth_per_dim;

CUDA_HOST_DEVICE
void CommonNetworkApi::set_event_queue(EventQueue* event_queue_ptr) noexcept {
    assert(event_queue_ptr != nullptr);
    CommonNetworkApi::event_queue = event_queue_ptr;
}

CUDA_HOST_DEVICE
CallbackTracker<1024>& CommonNetworkApi::get_callback_tracker() noexcept {
    return callback_tracker;
}

CUDA_HOST_DEVICE
void CommonNetworkApi::process_chunk_arrival(void* args) noexcept {
    assert(args != nullptr);

    // 解析块数据
    auto* const data = static_cast<FixedTuple<int, int, int, uint64_t, int>*>(args);
    const auto& [tag, src, dest, count, chunk_id] = *data;

    // 搜索跟踪器
    auto& tracker = CommonNetworkApi::get_callback_tracker();
    const auto entry = tracker.search_entry(tag, src, dest, count, chunk_id);
    assert(entry.has_value());  // 条目必须存在

    // 如果两个回调都已注册,则调用两个回调
    if (entry.value()->both_callbacks_registered()) {
        entry.value()->invoke_send_handler();
        entry.value()->invoke_recv_handler();

        // 移除条目
        tracker.pop_entry(tag, src, dest, count, chunk_id);
    } else {
        // 仅运行发送回调,因为接收尚未就绪
        entry.value()->invoke_send_handler();

        // 标记传输完成
        // 这样当调用sim_recv()时
        // 接收回调将立即被调用
        entry.value()->set_transmission_finished();
    }
}

CUDA_HOST_DEVICE
CommonNetworkApi::CommonNetworkApi(const int rank) noexcept : AstraNetworkAPI(rank) {
    assert(rank >= 0);
}

CUDA_HOST_DEVICE
TimeSpec CommonNetworkApi::sim_get_time() {
    // 从事件队列获取当前时间
    const auto current_time = event_queue->get_current_time();

    // 返回ASTRA-sim格式的当前时间
    const auto astra_sim_time = static_cast<double>(current_time);
    return {TimeType::NS, astra_sim_time};
}

CUDA_HOST_DEVICE
void CommonNetworkApi::sim_schedule(const TimeSpec delta, void (*fun_ptr)(void*), void* const fun_arg) {
    assert(delta.time_res == TimeType::NS);
    assert(fun_ptr != nullptr);

    // 计算绝对事件时间
    const auto current_time = sim_get_time();
    const auto event_time = current_time.time_val + delta.time_val;
    const auto event_time_ns = static_cast<EventTime>(event_time);

    // 将事件调度到事件队列
    assert(event_time_ns >= event_queue->get_current_time());
    event_queue->schedule_event(event_time_ns, fun_ptr, fun_arg);
}

CUDA_HOST_DEVICE
int CommonNetworkApi::sim_recv(
    void* const buffer,
    const uint64_t count,
    const int type,
    const int src,
    const int tag,
    SimRequest* const request,
    void (*msg_handler)(void*),
    void* const fun_arg) {
    // 查询块ID
    const auto dst = sim_comm_get_rank();
    const auto chunk_id = CommonNetworkApi::chunk_id_generator.create_recv_chunk_id(tag, src, dst, count);

    // 搜索跟踪器
    auto entry = callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // send()已调用
        // 行为取决于传输是否已完成
        if (entry.value()->is_transmission_finished()) {
            // 传输已完成,立即运行回调

            // 弹出条目
            callback_tracker.pop_entry(tag, src, dst, count, chunk_id);

            // 立即运行接收回调
            const auto delta = TimeSpec{TimeType::NS, 0};
            sim_schedule(delta, msg_handler, fun_arg);
        } else {
            // 传输尚未完成,只注册回调
            entry.value()->register_recv_callback(msg_handler, fun_arg);
        }
    } else {
        // send()尚未调用
        // 创建新条目并插入回调
        auto* const new_entry = callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_recv_callback(msg_handler, fun_arg);
    }

    // 返回
    return 0;
}

CUDA_HOST_DEVICE
double CommonNetworkApi::get_BW_at_dimension(const int dim) {
    assert(0 <= dim && dim < dims_count);

    // 返回请求维度的带宽
    return bandwidth_per_dim[dim];
}
