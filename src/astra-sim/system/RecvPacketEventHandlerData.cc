/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"

using namespace AstraSim;

CUDA_HOST_DEVICE
RecvPacketEventHandlerData::RecvPacketEventHandlerData() {
    this->wlhd = nullptr;
    this->owner = nullptr;
#ifndef __CUDA_ARCH__
    std::cout << "Created empty RecvPacketEventHandlerData" << std::endl;
#endif
}

CUDA_HOST_DEVICE
RecvPacketEventHandlerData::RecvPacketEventHandlerData(
    BaseStream* owner, int sys_id, EventType event, int vnet, int stream_id)
    : BasicEventHandlerData(sys_id, event) {
    this->workload = nullptr;
    this->wlhd = nullptr;
    this->owner = owner;
    this->vnet = vnet;
    this->stream_id = stream_id;
    this->message_end = true;
    ready_time = Sys::boostedTick();
#ifndef __CUDA_ARCH__
    std::cout << "Created RecvPacketEventHandlerData for sys_id: " << sys_id 
              << " vnet: " << vnet << " stream_id: " << stream_id << std::endl;
#endif
}
