/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "sys_layer/containers/FixedList.hpp"
#include "sys_layer/containers/FixedString.hpp"
#include "sys_layer/containers/FixedVector.hpp"

namespace AstraSim {

/**
 * 层数据类
 */
class LayerData {
  public:
    custom::FixedString layer_name;                                    ///< 层名称
    double total_forward_pass_compute;                                 ///< 前向传播计算总时间
    double total_weight_grad_compute;                                 ///< 权重梯度计算总时间
    double total_input_grad_compute;                                  ///< 输入梯度计算总时间
    double total_waiting_for_fwd_comm;                               ///< 等待前向通信总时间
    double total_waiting_for_wg_comm;                                ///< 等待权重梯度通信总时间
    double total_waiting_for_ig_comm;                                ///< 等待输入梯度通信总时间
    double total_fwd_comm;                                           ///< 前向通信总时间
    double total_weight_grad_comm;                                   ///< 权重梯度通信总时间
    double total_input_grad_comm;                                    ///< 输入梯度通信总时间
    custom::FixedList<std::pair<int, double>, 32> avg_queuing_delay;          ///< 平均排队延迟
    custom::FixedList<std::pair<int, double>, 32> avg_network_message_dealy;  ///< 平均网络消息延迟
};

/**
 * Astra仿真数据API类
 */
class AstraSimDataAPI {
  public:
    custom::FixedString run_name;                                     ///< 运行名称
    custom::FixedList<LayerData, 32> layers_stats;                   ///< 层统计数据
    custom::FixedVector<double, 32> avg_chunk_latency_per_logical_dimension;  ///< 每个逻辑维度的平均块延迟
    double workload_finished_time;                                    ///< 工作负载完成时间
    double total_compute;                                            ///< 总计算时间
    double total_exposed_comm;                                       ///< 总暴露通信时间
};

}  // namespace AstraSim
