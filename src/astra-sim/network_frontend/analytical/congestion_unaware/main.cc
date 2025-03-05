/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CmdLineParser.hh"
#include "congestion_unaware/CongestionUnawareNetworkApi.hh"
#include "sys_layer/containers/FixedVector.hpp"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_unaware/Helper.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>
#include <iostream>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionUnaware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

int main(int argc, char* argv[]) {
    std::cout << "Starting AstraSim analytical simulation..." << std::endl;
    
    // Parse command line arguments
    auto cmd_line_parser = CmdLineParser(argv[0]);
    cmd_line_parser.parse(argc, argv);

    // Get command line arguments
    const auto workload_configuration = cmd_line_parser.get<std::string>("workload-configuration");
    const auto comm_group_configuration = cmd_line_parser.get<std::string>("comm-group-configuration");
    const auto system_configuration = cmd_line_parser.get<std::string>("system-configuration");
    const auto remote_memory_configuration = cmd_line_parser.get<std::string>("remote-memory-configuration");
    const auto network_configuration = cmd_line_parser.get<std::string>("network-configuration");
    const auto num_queues_per_dim = cmd_line_parser.get<int>("num-queues-per-dim");
    const auto comm_scale = cmd_line_parser.get<double>("comm-scale");
    const auto injection_scale = cmd_line_parser.get<double>("injection-scale");
    const auto rendezvous_protocol = cmd_line_parser.get<bool>("rendezvous-protocol");

    std::cout << "Configurations loaded successfully" << std::endl;

    // Instantiate event queue
    const auto event_queue = std::make_shared<EventQueue>();
    std::cout << "Event queue created" << std::endl;

    // Generate topology
    const auto network_parser = NetworkParser(network_configuration);
    const auto topology = construct_topology(network_parser);
    std::cout << "Topology constructed" << std::endl;

    // Get topology information
    const auto npus_count = topology->get_npus_count();
    const auto npus_count_per_dim = topology->get_npus_count_per_dim();
    const auto dims_count = topology->get_dims_count();
    std::cout << "NPUs count: " << npus_count << ", Dims count: " << dims_count << std::endl;

    // Set up Network API
    CongestionUnawareNetworkApi::set_event_queue(event_queue);
    CongestionUnawareNetworkApi::set_topology(topology);
    std::cout << "Network API setup complete" << std::endl;

    // Create ASTRA-sim related resources
    auto network_apis = custom::FixedVector<std::unique_ptr<CongestionUnawareNetworkApi>, 128>();
    const auto memory_api = std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);
    auto systems = custom::FixedVector<Sys*, 128>();

    auto queues_per_dim = custom::FixedVector<int, 32>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(num_queues_per_dim);
    }
    std::cout << "Resource vectors initialized" << std::endl;

    for (int i = 0; i < npus_count; i++) {
        // create network and system
        auto network_api = std::make_unique<CongestionUnawareNetworkApi>(i);
        auto* const system = new Sys(i, workload_configuration, comm_group_configuration, system_configuration,
                                     memory_api.get(), network_api.get(), npus_count_per_dim, queues_per_dim,
                                     injection_scale, comm_scale, rendezvous_protocol);

        // push back network and system
        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
        std::cout << "Created system " << i << " of " << npus_count << std::endl;
    }

    // Initiate simulation
    std::cout << "Starting simulation..." << std::endl;
    for (int i = 0; i < npus_count; i++) {
        systems[i]->workload->fire();
        std::cout << "System " << i << " fired" << std::endl;
    }

    // run simulation
    int event_count = 0;
    while (!event_queue->finished()) {
        event_queue->proceed();
        event_count++;
        if (event_count % 1000 == 0) {
            std::cout << "Processed " << event_count << " events" << std::endl;
        }
    }

    std::cout << "Simulation completed with " << event_count << " events processed" << std::endl;
    return 0;
}
