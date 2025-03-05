/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/topology/ComplexLogicalTopology.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {

// 3D环形拓扑类
class Torus3D : public ComplexLogicalTopology {
public:
    CUDA_HOST_DEVICE Torus3D(int id, int total_nodes, int local_dim, int vertical_dim);
    CUDA_HOST_DEVICE ~Torus3D();

    CUDA_HOST_DEVICE int get_num_of_dimensions() override;
    CUDA_HOST_DEVICE int get_num_of_nodes_in_dimension(int dimension) override;
    CUDA_HOST_DEVICE BasicLogicalTopology* get_basic_topology_at_dimension(int dimension, ComType type) override;

    RingTopology* local_dimension;
    RingTopology* vertical_dimension;
    RingTopology* horizontal_dimension;
};

}  // namespace AstraSim
