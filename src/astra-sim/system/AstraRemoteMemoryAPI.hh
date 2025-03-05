/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdint>

namespace AstraSim {

class Sys;
class WorkloadLayerHandlerData;

/**
 * Astra远程内存API基类
 */
class AstraRemoteMemoryAPI {
  public:
    CUDA_HOST_DEVICE virtual ~AstraRemoteMemoryAPI() = default;
    
    /**
     * 设置系统实例
     */
    CUDA_HOST_DEVICE virtual void set_sys(int id, Sys* sys) = 0;
    
    /**
     * 发出内存请求
     */
    CUDA_HOST_DEVICE virtual void issue(uint64_t tensor_size, WorkloadLayerHandlerData* wlhd) = 0;
};

}  // namespace AstraSim
