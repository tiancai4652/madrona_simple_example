/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

/**
 * 可调用接口类
 */
class Callable {
  public:
    CUDA_HOST_DEVICE virtual ~Callable() = default;
    CUDA_HOST_DEVICE virtual void call(EventType type, CallData* data) = 0;
};

}  // namespace AstraSim
