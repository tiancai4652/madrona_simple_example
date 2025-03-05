/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/SharedBusStat.hh"
#include "astra-sim/system/Sys.hh"
#include "sys_layer/containers/FixedList.hpp"

namespace AstraSim {

class LogGP;

/**
 * 内存移动请求类
 */
class MemMovRequest : public Callable, public SharedBusStat {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    MemMovRequest(int request_num,
                  Sys* sys,
                  LogGP* loggp,
                  int size,
                  int latency,
                  Callable* callable,
                  bool processed,
                  bool send_back);

    /**
     * 等待内存总线
     */
    CUDA_HOST_DEVICE
    void wait_wait_for_mem_bus(typename custom::FixedList<MemMovRequest, 32>::iterator pointer);

    /**
     * 设置迭代器
     */
    CUDA_HOST_DEVICE
    void set_iterator(typename custom::FixedList<MemMovRequest, 32>::iterator pointer) {
        this->pointer = pointer;
    }

    /**
     * 调用处理
     */
    CUDA_HOST_DEVICE
    void call(EventType event, CallData* data) override;

    static int id;                    ///< 全局ID计数器
    int my_id;                       ///< 实例ID
    int size;                        ///< 大小
    int latency;                     ///< 延迟
    Callable* callable;              ///< 可调用对象
    bool processed;                  ///< 是否已处理
    bool send_back;                  ///< 是否需要发回
    bool mem_bus_finished;           ///< 内存总线是否完成
    Sys* sys;                       ///< 系统指针
    EventType callEvent;             ///< 调用事件类型
    LogGP* loggp;                   ///< LogGP指针
    typename custom::FixedList<MemMovRequest, 32>::iterator pointer;  ///< 迭代器

    Tick total_transfer_queue_time;   ///< 总传输队列时间
    Tick total_transfer_time;         ///< 总传输时间
    Tick total_processing_queue_time; ///< 总处理队列时间
    Tick total_processing_time;       ///< 总处理时间
    Tick start_time;                  ///< 开始时间
    int request_num;                  ///< 请求编号
};

}  // namespace AstraSim
