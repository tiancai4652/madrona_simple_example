/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <vector>

#include "astra-sim/system/topology/ComplexLogicalTopology.hh"

namespace AstraSim {

// 通用复杂拓扑类
class GeneralComplexTopology : public ComplexLogicalTopology {
  public:
    CUDA_HOST_DEVICE GeneralComplexTopology(int id, std::vector<int> dimension_size, std::vector<CollectiveImpl*> collective_impl);
    CUDA_HOST_DEVICE ~GeneralComplexTopology();

    CUDA_HOST_DEVICE int get_num_of_dimensions() override;
    CUDA_HOST_DEVICE int get_num_of_nodes_in_dimension(int dimension) override;
    CUDA_HOST_DEVICE BasicLogicalTopology* get_basic_topology_at_dimension(int dimension, ComType type) override;

    std::vector<LogicalTopology*> dimension_topology;
};

}  // namespace AstraSim
