/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/collective/Algorithm.hh"

using namespace AstraSim;

CUDA_HOST_DEVICE
Algorithm::Algorithm() {
    enabled = true;
}

CUDA_HOST_DEVICE
void Algorithm::init(BaseStream* stream) {
    this->stream = stream;
}

CUDA_HOST_DEVICE
void Algorithm::call(EventType event, CallData* data) {}

CUDA_HOST_DEVICE
void Algorithm::exit() {
    stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}
