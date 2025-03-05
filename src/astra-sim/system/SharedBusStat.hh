/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

/**
 * 共享总线统计类
 */
class SharedBusStat : public BasicEventHandlerData {
  public:
    CUDA_HOST_DEVICE SharedBusStat(BusType busType,
                  double total_bus_transfer_queue_delay,
                  double total_bus_transfer_delay,
                  double total_bus_processing_queue_delay,
                  double total_bus_processing_delay) {
#ifndef __CUDA_ARCH__
        std::cout << "Creating SharedBusStat for bus type " << static_cast<int>(busType) << std::endl;
#endif
        if (busType == BusType::Shared) {
            this->total_shared_bus_transfer_queue_delay = total_bus_transfer_queue_delay;
            this->total_shared_bus_transfer_delay = total_bus_transfer_delay;
            this->total_shared_bus_processing_queue_delay = total_bus_processing_queue_delay;
            this->total_shared_bus_processing_delay = total_bus_processing_delay;

            this->total_mem_bus_transfer_queue_delay = 0;
            this->total_mem_bus_transfer_delay = 0;
            this->total_mem_bus_processing_queue_delay = 0;
            this->total_mem_bus_processing_delay = 0;
        } else {
            this->total_shared_bus_transfer_queue_delay = 0;
            this->total_shared_bus_transfer_delay = 0;
            this->total_shared_bus_processing_queue_delay = 0;
            this->total_shared_bus_processing_delay = 0;

            this->total_mem_bus_transfer_queue_delay = total_bus_transfer_queue_delay;
            this->total_mem_bus_transfer_delay = total_bus_transfer_delay;
            this->total_mem_bus_processing_queue_delay = total_bus_processing_queue_delay;
            this->total_mem_bus_processing_delay = total_bus_processing_delay;
        }
        shared_request_counter = 0;
        mem_request_counter = 0;
    }

    CUDA_HOST_DEVICE void update_bus_stats(BusType busType, SharedBusStat* sharedBusStat) {
#ifndef __CUDA_ARCH__
        std::cout << "Updating bus stats for type " << static_cast<int>(busType) << std::endl;
#endif
        if (busType == BusType::Shared) {
            total_shared_bus_transfer_queue_delay += sharedBusStat->total_shared_bus_transfer_queue_delay;
            total_shared_bus_transfer_delay += sharedBusStat->total_shared_bus_transfer_delay;
            total_shared_bus_processing_queue_delay += sharedBusStat->total_shared_bus_processing_queue_delay;
            total_shared_bus_processing_delay += sharedBusStat->total_shared_bus_processing_delay;
            shared_request_counter++;
        } else if (busType == BusType::Mem) {
            total_mem_bus_transfer_queue_delay += sharedBusStat->total_mem_bus_transfer_queue_delay;
            total_mem_bus_transfer_delay += sharedBusStat->total_mem_bus_transfer_delay;
            total_mem_bus_processing_queue_delay += sharedBusStat->total_mem_bus_processing_queue_delay;
            total_mem_bus_processing_delay += sharedBusStat->total_mem_bus_processing_delay;
            mem_request_counter++;
        } else {
            total_shared_bus_transfer_queue_delay += sharedBusStat->total_shared_bus_transfer_queue_delay;
            total_shared_bus_transfer_delay += sharedBusStat->total_shared_bus_transfer_delay;
            total_shared_bus_processing_queue_delay += sharedBusStat->total_shared_bus_processing_queue_delay;
            total_shared_bus_processing_delay += sharedBusStat->total_shared_bus_processing_delay;
            total_mem_bus_transfer_queue_delay += sharedBusStat->total_mem_bus_transfer_queue_delay;
            total_mem_bus_transfer_delay += sharedBusStat->total_mem_bus_transfer_delay;
            total_mem_bus_processing_queue_delay += sharedBusStat->total_mem_bus_processing_queue_delay;
            total_mem_bus_processing_delay += sharedBusStat->total_mem_bus_processing_delay;
            shared_request_counter++;
            mem_request_counter++;
        }
    }

    CUDA_HOST_DEVICE void update_bus_stats(BusType busType, SharedBusStat sharedBusStat) {
#ifndef __CUDA_ARCH__
        std::cout << "Updating bus stats by value for type " << static_cast<int>(busType) << std::endl;
#endif
        if (busType == BusType::Shared) {
            total_shared_bus_transfer_queue_delay += sharedBusStat.total_shared_bus_transfer_queue_delay;
            total_shared_bus_transfer_delay += sharedBusStat.total_shared_bus_transfer_delay;
            total_shared_bus_processing_queue_delay += sharedBusStat.total_shared_bus_processing_queue_delay;
            total_shared_bus_processing_delay += sharedBusStat.total_shared_bus_processing_delay;
            shared_request_counter++;
        } else if (busType == BusType::Mem) {
            total_mem_bus_transfer_queue_delay += sharedBusStat.total_mem_bus_transfer_queue_delay;
            total_mem_bus_transfer_delay += sharedBusStat.total_mem_bus_transfer_delay;
            total_mem_bus_processing_queue_delay += sharedBusStat.total_mem_bus_processing_queue_delay;
            total_mem_bus_processing_delay += sharedBusStat.total_mem_bus_processing_delay;
            mem_request_counter++;
        } else {
            total_shared_bus_transfer_queue_delay += sharedBusStat.total_shared_bus_transfer_queue_delay;
            total_shared_bus_transfer_delay += sharedBusStat.total_shared_bus_transfer_delay;
            total_shared_bus_processing_queue_delay += sharedBusStat.total_shared_bus_processing_queue_delay;
            total_shared_bus_processing_delay += sharedBusStat.total_shared_bus_processing_delay;
            total_mem_bus_transfer_queue_delay += sharedBusStat.total_mem_bus_transfer_queue_delay;
            total_mem_bus_transfer_delay += sharedBusStat.total_mem_bus_transfer_delay;
            total_mem_bus_processing_queue_delay += sharedBusStat.total_mem_bus_processing_queue_delay;
            total_mem_bus_processing_delay += sharedBusStat.total_mem_bus_processing_delay;
            shared_request_counter++;
            mem_request_counter++;
        }
    }

    CUDA_HOST_DEVICE void take_bus_stats_average() {
#ifndef __CUDA_ARCH__
        std::cout << "Taking bus stats average" << std::endl;
#endif
        total_shared_bus_transfer_queue_delay /= shared_request_counter;
        total_shared_bus_transfer_delay /= shared_request_counter;
        total_shared_bus_processing_queue_delay /= shared_request_counter;
        total_shared_bus_processing_delay /= shared_request_counter;

        total_mem_bus_transfer_queue_delay /= mem_request_counter;
        total_mem_bus_transfer_delay /= mem_request_counter;
        total_mem_bus_processing_queue_delay /= mem_request_counter;
        total_mem_bus_processing_delay /= mem_request_counter;
    }

    double total_shared_bus_transfer_queue_delay;     ///< 共享总线传输队列延迟
    double total_shared_bus_transfer_delay;           ///< 共享总线传输延迟
    double total_shared_bus_processing_queue_delay;   ///< 共享总线处理队列延迟
    double total_shared_bus_processing_delay;         ///< 共享总线处理延迟

    double total_mem_bus_transfer_queue_delay;        ///< 内存总线传输队列延迟
    double total_mem_bus_transfer_delay;              ///< 内存总线传输延迟
    double total_mem_bus_processing_queue_delay;      ///< 内存总线处理队列延迟
    double total_mem_bus_processing_delay;            ///< 内存总线处理延迟
    int mem_request_counter;                          ///< 内存请求计数器
    int shared_request_counter;                       ///< 共享请求计数器
};

}  // namespace AstraSim
