/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

//
// Created by Saeed Rashidi on 7/7/22.
//

#include "astra-sim/system/CollectivePlan.hh"

namespace AstraSim {

CUDA_HOST_DEVICE
CollectivePlan::CollectivePlan(
    LogicalTopology* topology,
    custom::FixedVector<CollectiveImpl*, 32> implementation_per_dimension,
    custom::FixedVector<bool, 32> dimensions_involved,
    bool should_be_removed) {
    this->topology = topology;
    this->implementation_per_dimension = implementation_per_dimension;
    this->dimensions_involved = dimensions_involved;
    this->should_be_removed = should_be_removed;
#ifndef __CUDA_ARCH__
    std::cout << "Created CollectivePlan with topology=" << topology 
              << ", dimensions=" << dimensions_involved.size() << std::endl;
#endif
}

CUDA_HOST_DEVICE
CollectivePlan::~CollectivePlan() {
    if (should_be_removed) {
#ifndef __CUDA_ARCH__
        std::cout << "Destroying CollectivePlan and its components" << std::endl;
#endif
        delete topology;
        for (auto it = implementation_per_dimension.begin(); 
             it != implementation_per_dimension.end(); ++it) {
            delete *it;
        }
    }
}

} // namespace AstraSim
