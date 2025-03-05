/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/NetworkStat.hh"
#include "astra-sim/system/SharedBusStat.hh"
#include "sys_layer/containers/FixedList.hpp"

namespace AstraSim {

/**
 * 流统计类
 */
class StreamStat : public SharedBusStat, public NetworkStat {
  public:
    CUDA_HOST_DEVICE StreamStat() : SharedBusStat(BusType::Shared, 0, 0, 0, 0) {
        stream_stat_counter = 0;
#ifndef __CUDA_ARCH__
        std::cout << "Created StreamStat" << std::endl;
#endif
    }

    CUDA_HOST_DEVICE void update_stream_stats(StreamStat* streamStat) {
#ifndef __CUDA_ARCH__
        std::cout << "Updating stream stats" << std::endl;
#endif
        update_bus_stats(BusType::Both, streamStat);
        update_network_stat(streamStat);
        if (queuing_delay.size() < streamStat->queuing_delay.size()) {
            int dif = streamStat->queuing_delay.size() - queuing_delay.size();
            for (int i = 0; i < dif; i++) {
                queuing_delay.push_back(0);
            }
        }
        auto it = queuing_delay.begin();
        for (auto& tick : streamStat->queuing_delay) {
            (*it) += tick;
            ++it;
        }
        stream_stat_counter++;
    }

    CUDA_HOST_DEVICE void take_stream_stats_average() {
#ifndef __CUDA_ARCH__
        std::cout << "Taking stream stats average" << std::endl;
#endif
        take_bus_stats_average();
        take_network_stat_average();
        for (auto& tick : queuing_delay) {
            tick /= stream_stat_counter;
        }
    }

    custom::FixedList<double, 32> queuing_delay;  ///< 排队延迟列表
    int stream_stat_counter;                      ///< 流统计计数器
};

}  // namespace AstraSim
