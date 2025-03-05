/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/CollectivePlan.hh"
#include "astra-sim/system/Sys.hh"

namespace AstraSim {

CUDA_HOST_DEVICE
CommunicatorGroup::CommunicatorGroup(int id, custom::FixedVector<int, 32> involved_NPUs, Sys* generator) {
    set_id(id);
    this->involved_NPUs = involved_NPUs;
    this->generator = generator;
    custom::sort(involved_NPUs.begin(), involved_NPUs.end());
    assert(custom::find(involved_NPUs.begin(), involved_NPUs.end(), generator->id) != involved_NPUs.end());
#ifndef __CUDA_ARCH__
    std::cout << "Created CommunicatorGroup with id=" << id 
              << ", num_NPUs=" << involved_NPUs.size() << std::endl;
#endif
}

CUDA_HOST_DEVICE
CommunicatorGroup::~CommunicatorGroup() {
    for (auto it = comm_plans.begin(); it != comm_plans.end(); ++it) {
        CollectivePlan* cp = it->second;
        delete cp;
    }
}

CUDA_HOST_DEVICE
void CommunicatorGroup::set_id(int id) {
    assert(id > 0);
    this->id = id;
    this->num_streams = id * 1000000;
#ifndef __CUDA_ARCH__
    std::cout << "Set CommunicatorGroup id=" << id 
              << ", num_streams=" << num_streams << std::endl;
#endif
}

CUDA_HOST_DEVICE
CollectivePlan* CommunicatorGroup::get_collective_plan(ComType comm_type) {
    auto it = comm_plans.find(comm_type);
    if (it != comm_plans.end()) {
#ifndef __CUDA_ARCH__
        std::cout << "Found existing collective plan for comm_type=" 
                  << static_cast<int>(comm_type) << std::endl;
#endif
        return it->second;
    }

    if (static_cast<uint64_t>(generator->total_nodes) == involved_NPUs.size()) {
#ifndef __CUDA_ARCH__
        std::cout << "Creating new collective plan for all nodes, comm_type=" 
                  << static_cast<int>(comm_type) << std::endl;
#endif
        LogicalTopology* logical_topology = generator->get_logical_topology(comm_type);
        custom::FixedVector<CollectiveImpl*, 32> collective_implementation = 
            generator->get_collective_implementation(comm_type);
        custom::FixedVector<bool, 10> dimensions_involved(10, true);
        bool should_be_removed = false;
        comm_plans[comm_type] = new CollectivePlan(
            logical_topology, collective_implementation, dimensions_involved, should_be_removed);
        return comm_plans[comm_type];
    } else {
#ifndef __CUDA_ARCH__
        std::cout << "Creating new collective plan for subset of nodes, comm_type=" 
                  << static_cast<int>(comm_type) << std::endl;
#endif
        LogicalTopology* logical_topology = new RingTopology(
            RingTopology::Dimension::Local, generator->id, involved_NPUs);
        custom::FixedVector<CollectiveImpl*, 32> collective_implementation;
        collective_implementation.push_back(new CollectiveImpl(CollectiveImplType::Ring));
        custom::FixedVector<bool, 1> dimensions_involved(1, true);
        bool should_be_removed = true;
        comm_plans[comm_type] = new CollectivePlan(
            logical_topology, collective_implementation, dimensions_involved, should_be_removed);
        return comm_plans[comm_type];
    }
    assert(false);
    return nullptr;
}

} // namespace AstraSim
