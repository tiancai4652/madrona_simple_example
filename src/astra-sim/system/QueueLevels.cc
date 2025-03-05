/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/QueueLevels.hh"

using namespace AstraSim;

CUDA_HOST_DEVICE
QueueLevels::QueueLevels(int total_levels, int queues_per_level, int offset, AstraNetworkAPI::BackendType backend) {
    int start = offset;
    for (int i = 0; i < total_levels; i++) {
        QueueLevelHandler tmp(i, start, start + queues_per_level - 1, backend);
        levels.push_back(tmp);
#ifndef __CUDA_ARCH__
        std::cout << "Created queue level " << i << " with range [" << start 
                 << ", " << (start + queues_per_level - 1) << "]" << std::endl;
#endif
        start += queues_per_level;
    }
}

CUDA_HOST_DEVICE
QueueLevels::QueueLevels(custom::FixedVector<int, 8> lv, int offset, AstraNetworkAPI::BackendType backend) {
    int start = offset;
    int l = 0;
    for (int i = 0; i < lv.size(); i++) {
        QueueLevelHandler tmp(l, start, start + lv[i] - 1, backend);
        levels.push_back(tmp);
#ifndef __CUDA_ARCH__
        std::cout << "Created custom queue level " << l << " with range [" << start 
                 << ", " << (start + lv[i] - 1) << "]" << std::endl;
#endif
        start += lv[i];
        l++;
    }
}

CUDA_HOST_DEVICE
std::pair<int, RingTopology::Direction> QueueLevels::get_next_queue_at_level(int level) {
    auto result = levels[level].get_next_queue_id();
#ifndef __CUDA_ARCH__
    std::cout << "Getting next queue at level " << level << ": queue=" 
              << result.first << ", direction=" << static_cast<int>(result.second) << std::endl;
#endif
    return result;
}

CUDA_HOST_DEVICE
std::pair<int, RingTopology::Direction> QueueLevels::get_next_queue_at_level_first(int level) {
    auto result = levels[level].get_next_queue_id_first();
#ifndef __CUDA_ARCH__
    std::cout << "Getting first queue at level " << level << ": queue=" 
              << result.first << ", direction=" << static_cast<int>(result.second) << std::endl;
#endif
    return result;
}

CUDA_HOST_DEVICE
std::pair<int, RingTopology::Direction> QueueLevels::get_next_queue_at_level_last(int level) {
    auto result = levels[level].get_next_queue_id_last();
#ifndef __CUDA_ARCH__
    std::cout << "Getting last queue at level " << level << ": queue=" 
              << result.first << ", direction=" << static_cast<int>(result.second) << std::endl;
#endif
    return result;
}
