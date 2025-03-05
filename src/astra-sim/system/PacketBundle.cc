/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/PacketBundle.hh"

using namespace AstraSim;

CUDA_HOST_DEVICE
PacketBundle::PacketBundle(Sys* sys,
                           BaseStream* stream,
                           custom::FixedList<MyPacket*, 32> locked_packets,
                           bool needs_processing,
                           bool send_back,
                           uint64_t size,
                           MemBus::Transmition transmition) {
    this->sys = sys;
    this->locked_packets = locked_packets;
    this->needs_processing = needs_processing;
    this->send_back = send_back;
    this->size = size;
    this->stream = stream;
    this->transmition = transmition;
    creation_time = Sys::boostedTick();
#ifndef __CUDA_ARCH__
    std::cout << "Created PacketBundle with " << locked_packets.size() 
              << " packets, size=" << size << std::endl;
#endif
}

CUDA_HOST_DEVICE
PacketBundle::PacketBundle(Sys* sys,
                           BaseStream* stream,
                           bool needs_processing,
                           bool send_back,
                           uint64_t size,
                           MemBus::Transmition transmition) {
    this->sys = sys;
    this->needs_processing = needs_processing;
    this->send_back = send_back;
    this->size = size;
    this->stream = stream;
    this->transmition = transmition;
    creation_time = Sys::boostedTick();
#ifndef __CUDA_ARCH__
    std::cout << "Created empty PacketBundle with size=" << size << std::endl;
#endif
}

CUDA_HOST_DEVICE
void PacketBundle::send_to_MA() {
#ifndef __CUDA_ARCH__
    std::cout << "Sending PacketBundle to MA, size=" << size << std::endl;
#endif
    sys->memBus->send_from_NPU_to_MA(transmition, size, needs_processing, send_back, this);
}

CUDA_HOST_DEVICE
void PacketBundle::send_to_NPU() {
#ifndef __CUDA_ARCH__
    std::cout << "Sending PacketBundle to NPU, size=" << size << std::endl;
#endif
    sys->memBus->send_from_MA_to_NPU(transmition, size, needs_processing, send_back, this);
}

CUDA_HOST_DEVICE
void PacketBundle::call(EventType event, CallData* data) {
    if (needs_processing == true) {
        needs_processing = false;
        this->delay = static_cast<uint64_t>(static_cast<double>(size) / sys->local_mem_bw)     // write
                      + static_cast<uint64_t>(static_cast<double>(size) / sys->local_mem_bw)   // read
                      + static_cast<uint64_t>(static_cast<double>(size) / sys->local_mem_bw);  // read
#ifndef __CUDA_ARCH__
        std::cout << "Processing PacketBundle, delay=" << this->delay << std::endl;
#endif
        sys->try_register_event(this, EventType::CommProcessingFinished, data, this->delay);
        return;
    }
    Tick current = Sys::boostedTick();
    for (auto it = locked_packets.begin(); it != locked_packets.end(); ++it) {
        (*it)->ready_time = current;
    }
#ifndef __CUDA_ARCH__
    std::cout << "PacketBundle processing finished at time=" << current << std::endl;
#endif
    stream->call(EventType::General, data);
    delete this;
}
