/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "WorkloadLayerHandlerData.hh"

using namespace AstraSim;

CUDA_HOST_DEVICE
WorkloadLayerHandlerData::WorkloadLayerHandlerData() {
    node_id = 0;
    sys_id = 0;
    workload = nullptr;
}
