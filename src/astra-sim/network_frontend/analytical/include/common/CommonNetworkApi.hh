/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CallbackTracker.hh"
#include "common/ChunkIdGenerator.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-sim/system/AstraNetworkAPI.hh>
#include <astra-sim/system/Common.hh>
#include "sys_layer/containers/FixedVector.hpp"

using namespace AstraSim;
using namespace NetworkAnalytical;

namespace AstraSimAnalytical {

/**
 * CommonNetworkApi实现了通用的AstraNetworkAPI接口，
 * 该接口被拥塞感知和非拥塞感知的网络API继承。
 */
class CommonNetworkApi : public AstraNetworkAPI {
  public:
    /**
     * 设置要使用的事件队列
     */
    CUDA_HOST_DEVICE
    static void set_event_queue(EventQueue* event_queue_ptr) noexcept;

    /**
     * 获取回调跟踪器的引用
     */
    CUDA_HOST_DEVICE
    static CallbackTracker<1024>& get_callback_tracker() noexcept;

    /**
     * 当块到达目的地时要调用的回调
     */
    CUDA_HOST_DEVICE
    static void process_chunk_arrival(void* args) noexcept;

    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    explicit CommonNetworkApi(int rank) noexcept;

    /**
     * 实现AstraNetworkAPI的sim_get_time
     */
    CUDA_HOST_DEVICE
    [[nodiscard]] timespec_t sim_get_time() override;

    /**
     * 实现AstraNetworkAPI的sim_schedule
     */
    CUDA_HOST_DEVICE
    void sim_schedule(timespec_t delta, void (*fun_ptr)(void* fun_arg), void* fun_arg) override;

    /**
     * 实现AstraNetworkAPI的sim_recv
     */
    CUDA_HOST_DEVICE
    int sim_recv(void* buffer,
                 uint64_t count,
                 int type,
                 int src,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg) override;

    /**
     * 实现AstraNetworkAPI的get_BW_at_dimension
     */
    CUDA_HOST_DEVICE
    double get_BW_at_dimension(int dim) override;

  protected:
    /// 事件队列
    static EventQueue* event_queue;

    /// 块ID生成器
    static ChunkIdGenerator<1024> chunk_id_generator;

    /// 回调跟踪器
    static CallbackTracker<1024> callback_tracker;

    /// 拓扑中每个网络维度的带宽
    static custom::FixedVector<double, 8> bandwidth_per_dim;

    /// 拓扑的网络维度数量
    static int dims_count;
};

}  // namespace AstraSimAnalytical
