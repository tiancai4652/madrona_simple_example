/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MemMovRequest.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/Callable.hh"

namespace AstraSim {

class MemEventHandlerData : public Callable {
public:
    CUDA_HOST_DEVICE MemEventHandlerData(
        MemMovRequest* request,
        MemBus::Transmition transmition,
        bool send_back,
        bool processed,
        Callable* callable);

    CUDA_HOST_DEVICE ~MemEventHandlerData() override;
    CUDA_HOST_DEVICE void call(EventType event, CallData* data) override;

    MemMovRequest* request;
    MemBus::Transmition transmition;
    bool send_back;
    bool processed;
    Callable* callable;
};

} // namespace AstraSim
