/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

/**
 * 基础事件处理数据类
 */
class BasicEventHandlerData : public CallData {
public:
    CUDA_HOST_DEVICE BasicEventHandlerData();
    CUDA_HOST_DEVICE BasicEventHandlerData(int sys_id, EventType event);
    CUDA_HOST_DEVICE virtual ~BasicEventHandlerData() = default;

    int sys_id;      ///< 系统ID
    EventType event; ///< 事件类型
};

}  // namespace AstraSim
