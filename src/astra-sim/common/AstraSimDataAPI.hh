/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <string>
#include <vector>
#include "Common.hh"

namespace AstraSim {

// 层数据统计结构
struct alignas(8) LayerStats {
    std::string layer_name;
    
    // 计算时间统计
    double total_forward_pass_compute;
    double total_weight_grad_compute;
    double total_input_grad_compute;
    
    // 通信等待时间统计
    double total_waiting_for_fwd_comm;
    double total_waiting_for_wg_comm;
    double total_waiting_for_ig_comm;
    
    // 通信时间统计
    double total_fwd_comm;
    double total_weight_grad_comm;
    double total_input_grad_comm;
    
    // 延迟统计
    struct alignas(8) DelayStats {
        int phase;
        double latency;
    };
    std::vector<DelayStats> queuing_delays;     // 排队延迟统计
    std::vector<DelayStats> network_delays;     // 网络延迟统计

    CUDA_HOST_DEVICE LayerStats() :
        total_forward_pass_compute(0),
        total_weight_grad_compute(0),
        total_input_grad_compute(0),
        total_waiting_for_fwd_comm(0),
        total_waiting_for_wg_comm(0),
        total_waiting_for_ig_comm(0),
        total_fwd_comm(0),
        total_weight_grad_comm(0),
        total_input_grad_comm(0) {}
};

// 数据API类
class AstraSimDataAPI {
public:
    CUDA_HOST_DEVICE AstraSimDataAPI() :
        workload_finished_time(0),
        total_compute(0),
        total_exposed_comm(0) {}
        
    CUDA_HOST_DEVICE virtual ~AstraSimDataAPI() = default;

    std::string run_name;                                    // 运行名称
    std::vector<LayerStats> layers_stats;                   // 每层的统计数据
    std::vector<double> avg_chunk_latency_per_dimension;    // 每个逻辑维度的平均块延迟
    
    double workload_finished_time;                          // 工作负载完成时间
    double total_compute;                                   // 总计算时间
    double total_exposed_comm;                             // 总暴露通信时间
};

} // namespace AstraSim
