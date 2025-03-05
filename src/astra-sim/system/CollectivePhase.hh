/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class Algorithm;
class BaseStream;

/**
 * 集体通信阶段类
 */
class CollectivePhase {
public:
    CUDA_HOST_DEVICE CollectivePhase(Sys* sys, int queue_id, Algorithm* algorithm);
    CUDA_HOST_DEVICE CollectivePhase();
    CUDA_HOST_DEVICE void init(BaseStream* stream);

    Sys* sys;                ///< 系统指针
    int queue_id;            ///< 队列ID
    Algorithm* algorithm;     ///< 算法指针
    uint64_t initial_data_size;  ///< 初始数据大小
    uint64_t final_data_size;    ///< 最终数据大小
    bool enabled;            ///< 是否启用
    ComType comm_type;       ///< 通信类型
};

}  // namespace AstraSim
