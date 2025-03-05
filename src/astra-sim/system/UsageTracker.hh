/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdint>

#include "astra-sim/system/CSVWriter.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/Usage.hh"
#include "sys_layer/containers/FixedList.hpp"
#include "sys_layer/containers/FixedVector.hpp"

namespace AstraSim {

/**
 * 使用率追踪器类
 */
class UsageTracker {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    UsageTracker(int levels);

    /**
     * 增加使用率
     */
    CUDA_HOST_DEVICE
    void increase_usage();

    /**
     * 减少使用率
     */
    CUDA_HOST_DEVICE
    void decrease_usage();

    /**
     * 设置使用率
     */
    CUDA_HOST_DEVICE
    void set_usage(int level);

    /**
     * 生成报告
     */
    CUDA_HOST_DEVICE
    void report(CSVWriter* writer, int offset);

    /**
     * 生成百分比报告
     */
    CUDA_HOST_DEVICE
    custom::FixedList<std::pair<uint64_t, double>, 1024> report_percentage(uint64_t cycles);

    int levels;                                  ///< 级别数
    int current_level;                          ///< 当前级别
    Tick last_tick;                             ///< 上次时钟
    custom::FixedList<Usage, 1024> usage;       ///< 使用记录
};

}  // namespace AstraSim
