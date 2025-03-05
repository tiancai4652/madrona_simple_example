/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/BasicEventHandlerData.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/SimSendCaller.hh"
#include "astra-sim/system/Sys.hh"

namespace AstraSim {

/**
 * 会合发送数据类
 */
class RendezvousSendData : public BasicEventHandlerData, public MetaData {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    RendezvousSendData(int sys_id,
                       Sys* sys,
                       void* buffer,
                       uint64_t count,
                       int type,
                       int dst,
                       int tag,
                       sim_request request,
                       void (*msg_handler)(void* fun_arg),
                       void* fun_arg);

    SimSendCaller send;    ///< 发送调用器
};

}  // namespace AstraSim
