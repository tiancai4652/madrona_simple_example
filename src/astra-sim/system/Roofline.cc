/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/Roofline.hh"

#include <algorithm>

using namespace std;
using namespace AstraSim;

CUDA_HOST_DEVICE
Roofline::Roofline(double peak_perf) : peak_perf(peak_perf) {}

CUDA_HOST_DEVICE
Roofline::Roofline(double bandwidth, double peak_perf) : bandwidth(bandwidth), peak_perf(peak_perf) {}

CUDA_HOST_DEVICE
void Roofline::set_bandwidth(double bandwidth) {
    this->bandwidth = bandwidth;
}

CUDA_HOST_DEVICE
double Roofline::get_perf(double operational_intensity) {
#ifdef __CUDA_ARCH__
    return bandwidth * operational_intensity < peak_perf ? 
           bandwidth * operational_intensity : peak_perf;
#else
    return min(bandwidth * operational_intensity, peak_perf);
#endif
}
