/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/scheduling/OfflineGreedy.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace AstraSim;

// 静态成员初始化
custom::FixedMap<long long, custom::FixedVector<int, 8>, 1024> OfflineGreedy::chunk_schedule;
custom::FixedMap<long long, int, 1024> OfflineGreedy::schedule_consumer;
custom::FixedMap<long long, uint64_t, 1024> OfflineGreedy::global_chunk_size;

CUDA_HOST_DEVICE
DimElapsedTime::DimElapsedTime(int dim_num) {
    this->dim_num = dim_num;
    this->elapsed_time = 0;
}

CUDA_HOST_DEVICE
OfflineGreedy::OfflineGreedy(Sys* sys) {
    this->sys = sys;
    if (sys->dim_to_break == -1) {
        this->dim_size = sys->physical_dims;
        this->dim_BW.resize(this->dim_size.size());
        for (uint64_t i = 0; i < this->dim_size.size(); i++) {
            this->dim_BW[i] = sys->comm_NI->get_BW_at_dimension(i);
            this->dim_elapsed_time.push_back(DimElapsedTime(i));
        }
    } else {
        this->dim_size = sys->logical_broken_dims;
        this->dim_BW.resize(this->dim_size.size());
        for (uint64_t i = 0; i < this->dim_size.size(); i++) {
            if (i > static_cast<uint64_t>(sys->dim_to_break)) {
                this->dim_BW[i] = sys->comm_NI->get_BW_at_dimension(i - 1);
            } else {
                this->dim_BW[i] = sys->comm_NI->get_BW_at_dimension(i);
            }
            this->dim_elapsed_time.push_back(DimElapsedTime(i));
        }
    }
    if (sys->id == 0) {
        std::cout << "Themis is configured with the following parameters: " << std::endl;
        std::cout << "Dim size: ";
        for (uint64_t i = 0; i < this->dim_size.size(); i++) {
            std::cout << this->dim_size[i] << ", ";
        }
        std::cout << std::endl;
        std::cout << "BW per dim: ";
        for (uint64_t i = 0; i < this->dim_BW.size(); i++) {
            std::cout << this->dim_BW[i] << ", ";
        }
        std::cout << std::endl << std::endl;
    }
}

CUDA_HOST_DEVICE
void OfflineGreedy::reset_loads() {
    for (auto& dim : dim_elapsed_time) {
        dim.elapsed_time = 0;
    }
}

CUDA_HOST_DEVICE
uint64_t OfflineGreedy::get_chunk_size_from_elapsed_time(double elapsed_time, DimElapsedTime dim, ComType comm_type) {
    if (comm_type == ComType::Reduce_Scatter) {
        uint64_t result = ((elapsed_time * (dim_BW[dim.dim_num] / dim_BW[0])) /
                           (((double)(dim_size[dim.dim_num] - 1)) / (dim_size[dim.dim_num]))) *
                          1048576;
        return result;
    } else {
        uint64_t result =
            ((elapsed_time * (dim_BW[dim.dim_num] / dim_BW[0])) / (((double)(dim_size[dim.dim_num] - 1)) / (1))) *
            1048576;
        return result;
    }
}

CUDA_HOST_DEVICE
custom::FixedVector<int, 8> OfflineGreedy::get_chunk_scheduling(
    long long chunk_id,
    uint64_t& remaining_data_size,
    uint64_t recommended_chunk_size,
    custom::FixedVector<bool, 8>& dimensions_involved,
    InterDimensionScheduling inter_dim_scheduling,
    ComType comm_type) {
    
    if (chunk_schedule.find(chunk_id) != chunk_schedule.end()) {
        schedule_consumer[chunk_id]++;
        if (schedule_consumer[chunk_id] == static_cast<int64_t>(sys->all_sys.size())) {
            auto res = chunk_schedule[chunk_id];
            remaining_data_size -= global_chunk_size[chunk_id];
            chunk_schedule.erase(chunk_id);
            schedule_consumer.erase(chunk_id);
            global_chunk_size.erase(chunk_id);
            return res;
        }
        remaining_data_size -= global_chunk_size[chunk_id];
        return chunk_schedule[chunk_id];
    }

    if (sys->id != 0) {
        return sys->all_sys[0]->offline_greedy->get_chunk_scheduling(
            chunk_id, 
            remaining_data_size,
            recommended_chunk_size, 
            dimensions_involved,
            inter_dim_scheduling, 
            comm_type);
    }

    if (comm_type == ComType::All_Reduce) {
        comm_type = ComType::Reduce_Scatter;
    }

    std::sort(dim_elapsed_time.begin(), dim_elapsed_time.end());
    if (comm_type == ComType::All_Gather) {
        std::reverse(dim_elapsed_time.begin(), dim_elapsed_time.end());
    }

    custom::FixedVector<int, 8> result;
    uint64_t chunk_size = recommended_chunk_size;
    bool chunk_size_calculated = false;

    if (inter_dim_scheduling == InterDimensionScheduling::OfflineGreedy) {
        global_chunk_size[chunk_id] = std::min(remaining_data_size, chunk_size);
        remaining_data_size -= std::min(remaining_data_size, chunk_size);
    }

    for (auto& dim : dim_elapsed_time) {
        if (!dimensions_involved[dim.dim_num] || dim_size[dim.dim_num] == 1) {
            result.push_back(dim.dim_num);
            continue;
        }

        if (!chunk_size_calculated) {
            chunk_size_calculated = true;
            uint64_t diff_size = 0;

            if (comm_type == ComType::Reduce_Scatter) {
                double load_difference = fabs(dim_elapsed_time.back().elapsed_time - dim.elapsed_time);
                diff_size = get_chunk_size_from_elapsed_time(load_difference, dim, ComType::Reduce_Scatter);
            } else {
                int lastIndex = dim_elapsed_time.size() - 1;
                while (!dimensions_involved[dim_elapsed_time[lastIndex].dim_num] ||
                       dim_size[dim_elapsed_time[lastIndex].dim_num] == 1) {
                    lastIndex--;
                }
                double load_difference = fabs(dim_elapsed_time[lastIndex].elapsed_time - dim.elapsed_time);
                diff_size = get_chunk_size_from_elapsed_time(load_difference, dim_elapsed_time[lastIndex],
                                                             ComType::All_Gather);
            }

            if (diff_size < (recommended_chunk_size / 16)) {
                result.resize(dim_elapsed_time.size());
                for (uint64_t i = 0; i < dim_elapsed_time.size(); i++) {
                    result[i] = i;
                }
                chunk_schedule[chunk_id] = result;
                schedule_consumer[chunk_id] = 1;
                return result;
            }
        }

        result.push_back(dim.dim_num);
        if (comm_type == ComType::Reduce_Scatter) {
            dim.elapsed_time += ((((double)chunk_size) / 1048576) *
                                 (((double)(dim_size[dim.dim_num] - 1)) / (dim_size[dim.dim_num]))) /
                                (dim_BW[dim.dim_num] / dim_BW[0]);
            chunk_size /= dim_size[dim.dim_num];
        } else {
            dim.elapsed_time += ((((double)chunk_size) / 1048576) * (((double)(dim_size[dim.dim_num] - 1)))) /
                                (dim_BW[dim.dim_num] / dim_BW[0]);
            chunk_size *= dim_size[dim.dim_num];
        }
    }

    chunk_schedule[chunk_id] = result;
    schedule_consumer[chunk_id] = 1;
    return result;
}
