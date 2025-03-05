/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "sys_layer/containers/FixedConfig.hpp"
#include "sys_layer/containers/FixedString.hpp"
#include <cassert>

namespace AstraSimAnalytical {

/**
 * CmdLineParser使用FixedConfig解析ASTRA-sim的命令行参数。
 */
class CmdLineParser {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    CmdLineParser() noexcept;

    /**
     * 解析命令行参数
     */
    CUDA_HOST_DEVICE
    void parse_args(int argc, const char* const* argv) noexcept;

    /**
     * 获取工作负载配置文件路径
     */
    CUDA_HOST_DEVICE
    const char* get_workload_configuration() const noexcept;

    /**
     * 获取通信组配置文件路径
     */
    CUDA_HOST_DEVICE
    const char* get_comm_group_configuration() const noexcept;

    /**
     * 获取系统配置文件路径
     */
    CUDA_HOST_DEVICE
    const char* get_system_configuration() const noexcept;

    /**
     * 获取远程内存配置文件路径
     */
    CUDA_HOST_DEVICE
    const char* get_remote_memory_configuration() const noexcept;

    /**
     * 获取网络配置文件路径
     */
    CUDA_HOST_DEVICE
    const char* get_network_configuration() const noexcept;

    /**
     * 获取每个维度的队列数量
     */
    CUDA_HOST_DEVICE
    int get_num_queues_per_dim() const noexcept;

    /**
     * 获取计算缩放因子
     */
    CUDA_HOST_DEVICE
    double get_compute_scale() const noexcept;

    /**
     * 获取通信缩放因子
     */
    CUDA_HOST_DEVICE
    double get_comm_scale() const noexcept;

    /**
     * 获取注入缩放因子
     */
    CUDA_HOST_DEVICE
    double get_injection_scale() const noexcept;

    /**
     * 获取是否使用会合协议
     */
    CUDA_HOST_DEVICE
    bool get_rendezvous_protocol() const noexcept;

  private:
    /// 配置解析器
    custom::ConfigParser<32> config_parser_;
};

}  // namespace AstraSimAnalytical
