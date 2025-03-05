/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MemMovRequest.hh"
#include "astra-sim/system/LogGP.hh"
#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

int MemMovRequest::id = 0;

CUDA_HOST_DEVICE
MemMovRequest::MemMovRequest(
    int request_num, Sys* sys, LogGP* loggp, int size, int latency, Callable* callable, bool processed, bool send_back)
    : SharedBusStat(BusType::Mem, 0, 0, 0, 0) {
    this->size = size;
    this->latency = latency;
    this->callable = callable;
    this->processed = processed;
    this->send_back = send_back;
    this->my_id = id++;
    this->sys = sys;
    this->loggp = loggp;
    this->total_transfer_queue_time = 0;
    this->total_transfer_time = 0;
    this->total_processing_queue_time = 0;
    this->total_processing_time = 0;
    this->request_num = request_num;
    this->start_time = Sys::boostedTick();
    this->mem_bus_finished = true;
    this->callEvent = EventType::General;
#ifndef __CUDA_ARCH__
    std::cout << "Created MemMovRequest id=" << my_id << " size=" << size 
              << " latency=" << latency << std::endl;
#endif
}

CUDA_HOST_DEVICE
void MemMovRequest::call(EventType event, CallData* data) {
    update_bus_stats(BusType::Mem, (SharedBusStat*)data);
    total_mem_bus_transfer_delay += ((SharedBusStat*)data)->total_shared_bus_transfer_delay;
    total_mem_bus_processing_delay += ((SharedBusStat*)data)->total_shared_bus_processing_delay;
    total_mem_bus_processing_queue_delay += ((SharedBusStat*)data)->total_shared_bus_processing_queue_delay;
    total_mem_bus_transfer_queue_delay += ((SharedBusStat*)data)->total_shared_bus_transfer_queue_delay;
    mem_request_counter = 1;
    mem_bus_finished = true;
#ifndef __CUDA_ARCH__
    std::cout << "MemMovRequest id=" << my_id << " finished processing with delays: "
              << "transfer=" << total_mem_bus_transfer_delay 
              << " processing=" << total_mem_bus_processing_delay << std::endl;
#endif
    loggp->talking_it = pointer;
    loggp->call(callEvent, data);
}

CUDA_HOST_DEVICE
void MemMovRequest::wait_wait_for_mem_bus(typename custom::FixedList<MemMovRequest, 32>::iterator pointer) {
    mem_bus_finished = false;
    this->pointer = pointer;
#ifndef __CUDA_ARCH__
    std::cout << "MemMovRequest id=" << my_id << " waiting for memory bus" << std::endl;
#endif
}
