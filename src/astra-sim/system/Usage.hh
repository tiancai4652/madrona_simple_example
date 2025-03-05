/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdint>

namespace AstraSim {

/**
 * 使用情况类
 */
class Usage {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    Usage(int level, uint64_t start, uint64_t end);

    int level;          ///< 使用级别
    uint64_t start;     ///< 开始时间
    uint64_t end;       ///< 结束时间
};

}  // namespace AstraSim
