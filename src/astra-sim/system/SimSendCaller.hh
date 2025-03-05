/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Sys.hh"

namespace AstraSim {

/**
 * 模拟发送调用器类
 */
class SimSendCaller : public Callable {
  public:
    void* buffer;           ///< 发送缓冲区
    int count;             ///< 发送数据计数
    int type;              ///< 数据类型
    int dst;               ///< 目标节点
    int tag;               ///< 消息标签
    sim_request request;    ///< 模拟请求
    void (*msg_handler)(void* fun_arg);  ///< 消息处理函数
    void* fun_arg;         ///< 函数参数
    bool should_cleanup;    ///< 是否需要清理
    Sys* sys;              ///< 系统指针

    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    SimSendCaller(Sys* sys,
                  void* buffer,
                  int count,
                  int type,
                  int dst,
                  int tag,
                  sim_request request,
                  void (*msg_handler)(void* fun_arg),
                  void* fun_arg,
                  bool should_cleanup);

    /**
     * 调用函数
     */
    CUDA_HOST_DEVICE
    void call(EventType type, CallData* data) override;
};

}  // namespace AstraSim
