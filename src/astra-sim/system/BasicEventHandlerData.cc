/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

CUDA_HOST_DEVICE
BasicEventHandlerData::BasicEventHandlerData() {
    this->sys_id = 0;
#ifndef __CUDA_ARCH__
    std::cout << "Created empty BasicEventHandlerData" << std::endl;
#endif
}

CUDA_HOST_DEVICE
BasicEventHandlerData::BasicEventHandlerData(int sys_id, EventType event) {
    this->sys_id = sys_id;
    this->event = event;
#ifndef __CUDA_ARCH__
    std::cout << "Created BasicEventHandlerData with sys_id=" << sys_id 
              << ", event=" << static_cast<int>(event) << std::endl;
#endif
}

} // namespace AstraSim
