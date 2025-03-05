/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "sys_layer/containers/FixedString.hpp"

namespace AstraSim {

class Sys;
class LogGP;

/**
 * 内存总线类
 */
class MemBus {
  public:
    /**
     * 传输类型枚举
     */
    enum class Transmition { 
        Fast,    ///< 快速传输
        Usual    ///< 普通传输
    };

    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    MemBus(custom::FixedString side1,
           custom::FixedString side2,
           Sys* sys,
           Tick L,
           Tick o,
           Tick g,
           double G,
           bool model_shared_bus,
           int communication_delay,
           bool attach);

    /**
     * 析构函数
     */
    CUDA_HOST_DEVICE
    ~MemBus();

    /**
     * 从NPU发送到MA
     */
    CUDA_HOST_DEVICE
    void send_from_NPU_to_MA(Transmition transmition, int bytes, bool processed, bool send_back, Callable* callable);

    /**
     * 从MA发送到NPU
     */
    CUDA_HOST_DEVICE
    void send_from_MA_to_NPU(Transmition transmition, int bytes, bool processed, bool send_back, Callable* callable);

    LogGP* NPU_side;                ///< NPU端LogGP
    LogGP* MA_side;                 ///< MA端LogGP
    Sys* sys;                      ///< 系统指针
    int communication_delay;        ///< 通信延迟
    bool model_shared_bus;         ///< 是否模拟共享总线
};

}  // namespace AstraSim
