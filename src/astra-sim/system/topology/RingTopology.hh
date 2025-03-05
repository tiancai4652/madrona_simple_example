/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/system/topology/BasicLogicalTopology.hh"

namespace AstraSim {

// 环形拓扑类
class RingTopology : public BasicLogicalTopology {
public:
    enum class Direction { 
        Clockwise, 
        Anticlockwise 
    };
    
    enum class Dimension { 
        Local, 
        Vertical, 
        Horizontal, 
        NA 
    };

    CUDA_HOST_DEVICE int get_num_of_nodes_in_dimension(int dimension) override;
    
    CUDA_HOST_DEVICE RingTopology(Dimension dimension, int id, int total_nodes_in_ring, int index_in_ring, int offset);
    CUDA_HOST_DEVICE RingTopology(Dimension dimension, int id, std::vector<int> NPUs);
    
    CUDA_HOST_DEVICE virtual int get_receiver(int node_id, Direction direction);
    CUDA_HOST_DEVICE virtual int get_sender(int node_id, Direction direction);
    CUDA_HOST_DEVICE int get_nodes_in_ring();
    CUDA_HOST_DEVICE bool is_enabled();
    CUDA_HOST_DEVICE Dimension get_dimension();
    CUDA_HOST_DEVICE int get_index_in_ring();

private:
    std::unordered_map<int, int> id_to_index;
    std::unordered_map<int, int> index_to_id;

    std::string name;
    int id;
    int offset;
    int total_nodes_in_ring;
    int index_in_ring;
    Dimension dimension;

    CUDA_HOST_DEVICE virtual int get_receiver_homogeneous(int node_id, Direction direction, int offset);
};

} // namespace AstraSim
