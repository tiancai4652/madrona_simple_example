/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "Common.hh"

namespace AstraSim {

// 网络后端类型枚举
enum class BackendType { 
    NotSpecified = 0, 
    Garnet, 
    NS3, 
    Analytical 
};

// 网络API抽象基类
class AstraNetworkAPI {
public:
    CUDA_HOST_DEVICE AstraNetworkAPI(int rank) : rank(rank) {}
    CUDA_HOST_DEVICE virtual ~AstraNetworkAPI() = default;

    // 发送消息
    CUDA_HOST_DEVICE virtual int sim_send(
        void* buffer,
        uint64_t count,
        int type,
        int dst,
        int tag,
        SimRequest* request,
        void (*msg_handler)(void* fun_arg),
        void* fun_arg) = 0;

    // 接收消息
    CUDA_HOST_DEVICE virtual int sim_recv(
        void* buffer,
        uint64_t count,
        int type,
        int src,
        int tag,
        SimRequest* request,
        void (*msg_handler)(void* fun_arg),
        void* fun_arg) = 0;

    // 调度事件
    CUDA_HOST_DEVICE virtual void sim_schedule(
        TimeSpec delta, 
        void (*fun_ptr)(void* fun_arg), 
        void* fun_arg) = 0;

    // 获取后端类型
    CUDA_HOST_DEVICE virtual BackendType get_backend_type() {
        return BackendType::NotSpecified;
    }

    // 获取当前rank
    CUDA_HOST_DEVICE virtual int sim_comm_get_rank() {
        return rank;
    }

    // 设置当前rank
    CUDA_HOST_DEVICE virtual int sim_comm_set_rank(int rank) {
        this->rank = rank;
        return this->rank;
    }

    // 获取当前时间
    CUDA_HOST_DEVICE virtual TimeSpec sim_get_time() = 0;

    // 获取指定维度的带宽
    CUDA_HOST_DEVICE virtual double get_BW_at_dimension(int dim) {
        return -1;
    }

protected:
    int rank;
};

} // namespace AstraSim
