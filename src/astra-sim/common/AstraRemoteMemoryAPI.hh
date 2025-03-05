/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdint>
#include "Common.hh"

namespace AstraSim {

// 前向声明
class Sys;
class WorkloadLayerHandlerData;

// 远程内存API抽象基类
class AstraRemoteMemoryAPI {
public:
    CUDA_HOST_DEVICE AstraRemoteMemoryAPI() = default;
    CUDA_HOST_DEVICE virtual ~AstraRemoteMemoryAPI() = default;

    // 设置系统实例
    CUDA_HOST_DEVICE virtual void set_sys(int id, Sys* sys) = 0;

    // 发起远程内存请求
    CUDA_HOST_DEVICE virtual void issue(
        uint64_t tensor_size,
        WorkloadLayerHandlerData* wlhd) = 0;
};

}  // namespace AstraSim
