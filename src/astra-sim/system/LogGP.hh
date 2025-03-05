/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MemMovRequest.hh"
#include "sys_layer/containers/FixedList.hpp"
#include "sys_layer/containers/FixedString.hpp"

namespace AstraSim {

class Sys;

/**
 * LogGP模型类，用于模拟网络通信
 */
class LogGP : public Callable {
public:
    CUDA_HOST_DEVICE LogGP(custom::FixedString name, Sys* sys, Tick L, Tick o, Tick g, double G, EventType trigger_event);
    CUDA_HOST_DEVICE ~LogGP();
    CUDA_HOST_DEVICE void process_next_read();
    CUDA_HOST_DEVICE void request_read(int bytes, bool processed, bool send_back, Callable* callable);
    CUDA_HOST_DEVICE void switch_to_receiver(MemMovRequest mr, Tick offset);
    CUDA_HOST_DEVICE void call(EventType event, CallData* data) override;
    CUDA_HOST_DEVICE void attach_mem_bus(Sys* sys, Tick L, Tick o, Tick g, double G, bool model_shared_bus, int communication_delay);

    enum class State { Free, waiting, Sending, Receiving };  ///< 状态枚举
    enum class ProcState { Free, Processing };              ///< 处理状态枚举

    MemBus* NPU_MEM;                                       ///< NPU内存总线
    int request_num;                                       ///< 请求数量
    custom::FixedString name;                              ///< 名称
    Tick L;                                               ///< 延迟参数
    Tick o;                                               ///< 开销参数
    Tick g;                                               ///< 间隔参数
    double G;                                             ///< 带宽参数
    Tick last_trans;                                      ///< 上次传输时间
    State curState;                                       ///< 当前状态
    State prevState;                                      ///< 前一状态
    ProcState processing_state;                           ///< 处理状态

    custom::FixedList<MemMovRequest, 32> sends;           ///< 发送队列
    custom::FixedList<MemMovRequest, 32> receives;        ///< 接收队列
    custom::FixedList<MemMovRequest, 32> processing;      ///< 处理队列
    custom::FixedList<MemMovRequest, 32> retirements;     ///< 退休队列
    custom::FixedList<MemMovRequest, 32> pre_send;        ///< 预发送队列
    custom::FixedList<MemMovRequest, 32> pre_process;     ///< 预处理队列
    typename custom::FixedList<MemMovRequest, 32>::iterator talking_it;  ///< 当前处理的迭代器

    LogGP* partner;                                       ///< 伙伴节点
    Sys* sys;                                            ///< 系统指针
    EventType trigger_event;                             ///< 触发事件
    int subsequent_reads;                                ///< 后续读取数
    int THRESHOLD;                                       ///< 阈值
    int local_reduction_delay;                           ///< 本地规约延迟
};

}  // namespace AstraSim
