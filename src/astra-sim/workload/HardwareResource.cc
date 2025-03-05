/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// TODO: HardwareResource.cc should be moved to the system layer.

#include "astra-sim/workload/HardwareResource.hh"
#include <cassert>

using namespace std;
using namespace AstraSim;
using namespace Chakra;

typedef ChakraProtoMsg::NodeType ChakraNodeType;

CUDA_HOST_DEVICE
HardwareResource::HardwareResource(uint32_t num_npus)
    : num_npus(num_npus),
      num_in_flight_cpu_ops(0),
      num_in_flight_gpu_comp_ops(0),
      num_in_flight_gpu_comm_ops(0),
      peak_performance(1000.0f),  // 默认1 TFLOPS
      memory_bandwidth(900.0f)    // 默认900 GB/s
{}

CUDA_HOST_DEVICE
void HardwareResource::occupy(const std::shared_ptr<Chakra::ETFeederNode> node) {
    if (node->is_cpu_op()) {
        assert(num_in_flight_cpu_ops == 0);
        ++num_in_flight_cpu_ops;
    } else {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            assert(num_in_flight_gpu_comp_ops == 0);
            ++num_in_flight_gpu_comp_ops;
        } else {
            assert(num_in_flight_gpu_comm_ops == 0);
            ++num_in_flight_gpu_comm_ops;
        }
    }
}

CUDA_HOST_DEVICE
void HardwareResource::release(const std::shared_ptr<Chakra::ETFeederNode> node) {
    if (node->is_cpu_op()) {
        --num_in_flight_cpu_ops;
        assert(num_in_flight_cpu_ops == 0);
    } else {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            --num_in_flight_gpu_comp_ops;
            assert(num_in_flight_gpu_comp_ops == 0);
        } else {
            --num_in_flight_gpu_comm_ops;
            assert(num_in_flight_gpu_comm_ops == 0);
        }
    }
}

CUDA_HOST_DEVICE
bool HardwareResource::is_available(const std::shared_ptr<Chakra::ETFeederNode> node) const {
    if (node->is_cpu_op()) {
        return num_in_flight_cpu_ops == 0;
    } else {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            return num_in_flight_gpu_comp_ops == 0;
        } else {
            return num_in_flight_gpu_comm_ops == 0;
        }
    }
}
