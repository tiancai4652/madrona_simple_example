/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/StreamBaseline.hh"

namespace AstraSim {

custom::FixedMap<int, int, 128> BaseStream::synchronizer;
custom::FixedMap<int, int, 128> BaseStream::ready_counter;
custom::FixedMap<int, custom::FixedList<BaseStream*, 32>, 128> BaseStream::suspended_streams;

CUDA_HOST_DEVICE
void BaseStream::changeState(StreamState state) {
    this->state = state;
#ifndef __CUDA_ARCH__
    std::cout << "Stream " << stream_id << " changed state to " 
              << static_cast<int>(state) << std::endl;
#endif
}

CUDA_HOST_DEVICE
BaseStream::BaseStream(int stream_id, Sys* owner, custom::FixedList<CollectivePhase, 32> phases_to_go) {
    this->stream_id = stream_id;
    this->owner = owner;
    this->initialized = false;
    this->phases_to_go = phases_to_go;
    
#ifndef __CUDA_ARCH__
    std::cout << "Creating BaseStream with id=" << stream_id << std::endl;
#endif

    if (synchronizer.find(stream_id) != synchronizer.end()) {
        synchronizer[stream_id]++;
#ifndef __CUDA_ARCH__
        std::cout << "Incrementing synchronizer for stream " << stream_id 
                  << " to " << synchronizer[stream_id] << std::endl;
#endif
    } else {
        synchronizer[stream_id] = 1;
        ready_counter[stream_id] = 0;
#ifndef __CUDA_ARCH__
        std::cout << "Initializing synchronizer for stream " << stream_id << std::endl;
#endif
    }

    for (auto& vn : phases_to_go) {
        if (vn.algorithm != nullptr) {
            vn.init(this);
        }
    }

    state = StreamState::Created;
    preferred_scheduling = SchedulingPolicy::None;
    creation_time = Sys::boostedTick();
    total_packets_sent = 0;
    current_queue_id = -1;
    priority = 0;

#ifndef __CUDA_ARCH__
    std::cout << "BaseStream " << stream_id << " initialization complete at time " 
              << creation_time << std::endl;
#endif
}

} // namespace AstraSim
