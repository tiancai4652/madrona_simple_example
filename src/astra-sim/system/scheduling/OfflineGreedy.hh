/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <vector>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/Sys.hh"
#include "sys_layer/containers/FixedVector.hpp"
#include "sys_layer/containers/FixedMap.hpp"

namespace AstraSim {

/**
 * 维度经过时间类
 */
class DimElapsedTime {
  public:
    int dim_num;                ///< 维度编号
    double elapsed_time;        ///< 经过时间

    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    DimElapsedTime(int dim_num);

    /**
     * 比较运算符
     */
    CUDA_HOST_DEVICE
    bool operator<(const DimElapsedTime& dimElapsedTime) const {
        return (elapsed_time < dimElapsedTime.elapsed_time);
    }
};

/**
 * 离线贪心调度器
 */
class OfflineGreedy {
  public:
    Sys* sys;                                  ///< 系统指针
    custom::FixedVector<DimElapsedTime, 8> dim_elapsed_time;  ///< 维度经过时间
    custom::FixedVector<double, 8> dim_BW;     ///< 维度带宽
    custom::FixedVector<int, 8> dim_size;      ///< 维度大小

    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    OfflineGreedy(Sys* sys);

    /**
     * 重置负载
     */
    CUDA_HOST_DEVICE
    void reset_loads();

    /**
     * 获取分块调度
     */
    CUDA_HOST_DEVICE
    custom::FixedVector<int, 8> get_chunk_scheduling(
        long long chunk_id,
        uint64_t& remaining_data_size,
        uint64_t recommended_chunk_size,
        custom::FixedVector<bool, 8>& dimensions_involved,
        InterDimensionScheduling inter_dim_scheduling,
        ComType comm_type);

    /**
     * 从经过时间获取分块大小
     */
    CUDA_HOST_DEVICE
    uint64_t get_chunk_size_from_elapsed_time(double elapsed_time, DimElapsedTime dim, ComType comm_type);

    static custom::FixedMap<long long, custom::FixedVector<int, 8>, 1024> chunk_schedule;  ///< 分块调度表
    static custom::FixedMap<long long, int, 1024> schedule_consumer;                       ///< 调度消费者
    static custom::FixedMap<long long, uint64_t, 1024> global_chunk_size;                 ///< 全局分块大小
};

}  // namespace AstraSim
