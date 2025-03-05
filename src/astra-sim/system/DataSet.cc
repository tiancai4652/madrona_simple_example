/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/DataSet.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/Sys.hh"

namespace AstraSim {

int DataSet::id_auto_increment = 0;

CUDA_HOST_DEVICE
DataSet::DataSet(int total_streams) {
    this->my_id = id_auto_increment++;
    this->total_streams = total_streams;
    this->finished_streams = 0;
    this->finished = false;
    this->finish_tick = 0;
    this->active = true;
    this->creation_tick = Sys::boostedTick();
    this->notifier = nullptr;
#ifndef __CUDA_ARCH__
    std::cout << "Created DataSet with ID=" << my_id 
              << ", total_streams=" << total_streams << std::endl;
#endif
}

CUDA_HOST_DEVICE
void DataSet::set_notifier(Callable* callable, EventType event) {
#ifndef __CUDA_ARCH__
    std::cout << "Setting notifier for DataSet " << my_id << std::endl;
#endif
    notifier = new custom::FixedTuple<Callable*, EventType>(callable, event);
}

CUDA_HOST_DEVICE
void DataSet::notify_stream_finished(StreamStat* data) {
    finished_streams++;
#ifndef __CUDA_ARCH__
    std::cout << "Stream finished for DataSet " << my_id 
              << ", finished_streams=" << finished_streams 
              << "/" << total_streams << std::endl;
#endif
    if (data != nullptr) {
        update_stream_stats(data);
    }
    if (finished_streams == total_streams) {
        finished = true;
        finish_tick = Sys::boostedTick();
        if (notifier != nullptr) {
            take_stream_stats_average();
            Callable* c = custom::get<0>(*notifier);
            EventType ev = custom::get<1>(*notifier);
            delete notifier;
            c->call(ev, new IntData(my_id));
#ifndef __CUDA_ARCH__
            std::cout << "All streams finished for DataSet " << my_id 
                      << ", notifying with event=" << static_cast<int>(ev) << std::endl;
#endif
        }
    }
}

CUDA_HOST_DEVICE
void DataSet::call(EventType event, CallData* data) {
#ifndef __CUDA_ARCH__
    std::cout << "DataSet " << my_id << " received event " 
              << static_cast<int>(event) << std::endl;
#endif
    notify_stream_finished(((StreamStat*)data));
}

CUDA_HOST_DEVICE
bool DataSet::is_finished() {
    return finished;
}

} // namespace AstraSim
