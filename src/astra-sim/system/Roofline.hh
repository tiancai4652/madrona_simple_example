/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

namespace AstraSim {

/**
 * Roofline性能模型类
 */
class Roofline {
  public:
    /**
     * 构造函数 - 仅峰值性能
     */
    CUDA_HOST_DEVICE
    Roofline(double peak_perf);

    /**
     * 构造函数 - 带宽和峰值性能
     */
    CUDA_HOST_DEVICE
    Roofline(double bandwidth, double peak_perf);

    /**
     * 设置带宽
     */
    CUDA_HOST_DEVICE
    void set_bandwidth(double bandwidth);

    /**
     * 获取性能
     */
    CUDA_HOST_DEVICE
    double get_perf(double operational_intensity);

  private:
    double bandwidth;    ///< 带宽
    double peak_perf;    ///< 峰值性能
};

}  // namespace AstraSim
