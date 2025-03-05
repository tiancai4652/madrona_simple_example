/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/SendPacketEventHandlerData.hh"

using namespace AstraSim;

CUDA_HOST_DEVICE
SendPacketEventHandlerData::SendPacketEventHandlerData() {
    tag = 0;
    callable = nullptr;
    wlhd = nullptr;
}

CUDA_HOST_DEVICE
SendPacketEventHandlerData::SendPacketEventHandlerData(Callable* callable, int tag)
    : BasicEventHandlerData(-1, EventType::PacketSent) {
    this->callable = callable;
    this->tag = tag;
}
