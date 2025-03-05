/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MemEventHandlerData.hh"

namespace AstraSim {

CUDA_HOST_DEVICE
MemEventHandlerData::MemEventHandlerData(
    MemMovRequest* request,
    MemBus::Transmition transmition,
    bool send_back,
    bool processed,
    Callable* callable)
    : request(request),
      transmition(transmition),
      send_back(send_back),
      processed(processed),
      callable(callable) {
#ifndef __CUDA_ARCH__
    std::cout << "Created MemEventHandlerData with request=" << request 
              << ", send_back=" << send_back << ", processed=" << processed << std::endl;
#endif
}

CUDA_HOST_DEVICE
MemEventHandlerData::~MemEventHandlerData() {
    // No need to delete request or callable as they are managed elsewhere
}

CUDA_HOST_DEVICE
void MemEventHandlerData::call(EventType event, CallData* data) {
#ifndef __CUDA_ARCH__
    std::cout << "MemEventHandlerData::call - Event: " << static_cast<int>(event) << std::endl;
#endif
    if (callable) {
        callable->call(event, data);
    }
#ifndef __CUDA_ARCH__
    else {
        std::cout << "Warning: MemEventHandlerData::call - callable is null" << std::endl;
    }
#endif
}

} // namespace AstraSim
