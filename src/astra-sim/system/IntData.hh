/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/CallData.hh"

namespace AstraSim {

/**
 * 整数数据类
 */
class IntData : public CallData {
public:
    CUDA_HOST_DEVICE IntData(int d) {
        data = d;
#ifndef __CUDA_ARCH__
        std::cout << "Created IntData with value=" << d << std::endl;
#endif
    }
    
    int data;  ///< 整数数据
};

}  // namespace AstraSim
