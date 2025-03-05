/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Common.hh"
#include "sys_layer/containers/FixedList.hpp"

namespace AstraSim {

/**
 * 网络统计类
 */
class NetworkStat {
public:
    CUDA_HOST_DEVICE NetworkStat() {
        net_message_counter = 0;
#ifndef __CUDA_ARCH__
        std::cout << "Created NetworkStat" << std::endl;
#endif
    }

    CUDA_HOST_DEVICE void update_network_stat(NetworkStat* networkStat) {
        if (net_message_latency.size() < networkStat->net_message_latency.size()) {
            int dif = networkStat->net_message_latency.size() - net_message_latency.size();
            for (int i = 0; i < dif; i++) {
                net_message_latency.push_back(0);
            }
        }
#ifndef __CUDA_ARCH__
        std::cout << "Updating network stats, message count=" << net_message_counter << std::endl;
#endif
        auto it = net_message_latency.begin();
        for (auto& ml : networkStat->net_message_latency) {
            (*it) += ml;
            ++it;
        }
        net_message_counter++;
    }

    CUDA_HOST_DEVICE void take_network_stat_average() {
#ifndef __CUDA_ARCH__
        std::cout << "Taking network stats average for " << net_message_counter << " messages" << std::endl;
#endif
        for (auto& ml : net_message_latency) {
            ml /= net_message_counter;
        }
    }

    custom::FixedList<double, 32> net_message_latency;  ///< 消息延迟列表
    int net_message_counter;                            ///< 消息计数器
};

}  // namespace AstraSim
