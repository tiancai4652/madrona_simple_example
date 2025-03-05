/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MyPacket.hh"

using namespace AstraSim;

CUDA_HOST_DEVICE
MyPacket::MyPacket(int preferred_vnet, int preferred_src, int preferred_dest) {
    this->preferred_vnet = preferred_vnet;
    this->preferred_src = preferred_src;
    this->preferred_dest = preferred_dest;
    this->msg_size = 0;
#ifndef __CUDA_ARCH__
    std::cout << "Created MyPacket with vnet=" << preferred_vnet 
              << " src=" << preferred_src << " dest=" << preferred_dest << std::endl;
#endif
}

CUDA_HOST_DEVICE
MyPacket::MyPacket(uint64_t msg_size, int preferred_vnet, int preferred_src, int preferred_dest) {
    this->preferred_vnet = preferred_vnet;
    this->preferred_src = preferred_src;
    this->preferred_dest = preferred_dest;
    this->msg_size = msg_size;
#ifndef __CUDA_ARCH__
    std::cout << "Created MyPacket with size=" << msg_size << " vnet=" << preferred_vnet 
              << " src=" << preferred_src << " dest=" << preferred_dest << std::endl;
#endif
}

CUDA_HOST_DEVICE
void MyPacket::set_notifier(Callable* c) {
    notifier = c;
#ifndef __CUDA_ARCH__
    std::cout << "Set notifier for packet with vnet=" << preferred_vnet << std::endl;
#endif
}

CUDA_HOST_DEVICE
void MyPacket::call(EventType event, CallData* data) {
    cycles_needed = 0;
    if (notifier != nullptr) {
#ifndef __CUDA_ARCH__
        std::cout << "Calling notifier for packet with vnet=" << preferred_vnet << std::endl;
#endif
        notifier->call(EventType::General, nullptr);
    }
}
