/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CmdLineParser.hh"
#include "sys_layer/containers/FixedConfig.hpp"

using namespace AstraSimAnalytical;

CUDA_HOST_DEVICE
CmdLineParser::CmdLineParser() noexcept {
    // 初始化配置解析器
    config_parser_.add_string_option("workload-configuration", "", true);
    config_parser_.add_string_option("comm-group-configuration", "empty", false);
    config_parser_.add_string_option("system-configuration", "", true);
    config_parser_.add_string_option("remote-memory-configuration", "", true);
    config_parser_.add_string_option("network-configuration", "", true);
    config_parser_.add_int_option("num-queues-per-dim", 1);
    config_parser_.add_double_option("compute-scale", 1.0);
    config_parser_.add_double_option("comm-scale", 1.0);
    config_parser_.add_double_option("injection-scale", 1.0);
    config_parser_.add_bool_option("rendezvous-protocol", false);
}

CUDA_HOST_DEVICE
void CmdLineParser::parse_args(int argc, const char* const* argv) noexcept {
    // 解析参数
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            break;
        }

        const char* name = argv[i];
        const char* value = argv[i + 1];

        // 跳过选项名称中的前导'-'
        while (*name == '-') {
            name++;
        }

        // 根据选项类型设置值
        if (!config_parser_.set_string_value(name, value)) {
            if (!config_parser_.set_int_value(name, atoi(value))) {
                if (!config_parser_.set_double_value(name, atof(value))) {
                    if (strcmp(value, "true") == 0) {
                        config_parser_.set_bool_value(name, true);
                    } else if (strcmp(value, "false") == 0) {
                        config_parser_.set_bool_value(name, false);
                    }
                }
            }
        }
    }

    // 验证必需选项
    assert(config_parser_.validate());
}

CUDA_HOST_DEVICE
const char* CmdLineParser::get_workload_configuration() const noexcept {
    return config_parser_.get_string_value("workload-configuration").c_str();
}

CUDA_HOST_DEVICE
const char* CmdLineParser::get_comm_group_configuration() const noexcept {
    return config_parser_.get_string_value("comm-group-configuration").c_str();
}

CUDA_HOST_DEVICE
const char* CmdLineParser::get_system_configuration() const noexcept {
    return config_parser_.get_string_value("system-configuration").c_str();
}

CUDA_HOST_DEVICE
const char* CmdLineParser::get_remote_memory_configuration() const noexcept {
    return config_parser_.get_string_value("remote-memory-configuration").c_str();
}

CUDA_HOST_DEVICE
const char* CmdLineParser::get_network_configuration() const noexcept {
    return config_parser_.get_string_value("network-configuration").c_str();
}

CUDA_HOST_DEVICE
int CmdLineParser::get_num_queues_per_dim() const noexcept {
    return config_parser_.get_int_value("num-queues-per-dim");
}

CUDA_HOST_DEVICE
double CmdLineParser::get_compute_scale() const noexcept {
    return config_parser_.get_double_value("compute-scale");
}

CUDA_HOST_DEVICE
double CmdLineParser::get_comm_scale() const noexcept {
    return config_parser_.get_double_value("comm-scale");
}

CUDA_HOST_DEVICE
double CmdLineParser::get_injection_scale() const noexcept {
    return config_parser_.get_double_value("injection-scale");
}

CUDA_HOST_DEVICE
bool CmdLineParser::get_rendezvous_protocol() const noexcept {
    return config_parser_.get_bool_value("rendezvous-protocol");
}
