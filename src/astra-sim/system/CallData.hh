/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

namespace AstraSim {

/**
 * 调用数据基类
 */
class CallData {
public:
    CUDA_HOST_DEVICE virtual ~CallData() = default;
};

}  // namespace AstraSim
