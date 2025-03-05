/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CommonNetworkApi.hh"
#include "sys_layer/containers/FixedVector.hpp"
#include <astra-network-analytical/common/Type.h>
#include <astra-network-analytical/congestion_unaware/Topology.h>

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

namespace AstraSimAnalyticalCongestionUnaware {

/**
 * 无拥塞感知网络API类
 * 用于无拥塞分析网络后端的AstraNetworkAPI实现
 */
class CongestionUnawareNetworkApi final : public CommonNetworkApi {
  public:
    /**
     * 设置要使用的拓扑
     * @param topology_ptr 拓扑指针
     */
    CUDA_HOST_DEVICE static void set_topology(std::shared_ptr<Topology> topology_ptr) noexcept {
#ifndef __CUDA_ARCH__
        std::cout << "Setting topology for CongestionUnawareNetworkApi" << std::endl;
#endif
        topology = topology_ptr;
    }

    /**
     * 构造函数
     * @param rank API的ID
     */
    CUDA_HOST_DEVICE explicit CongestionUnawareNetworkApi(int rank) noexcept : CommonNetworkApi(rank) {
#ifndef __CUDA_ARCH__
        std::cout << "Created CongestionUnawareNetworkApi with rank=" << rank << std::endl;
#endif
    }

    /**
     * 实现AstraNetworkAPI的sim_send方法
     */
    CUDA_HOST_DEVICE int sim_send(void* buffer,
                 uint64_t count,
                 int type,
                 int dst,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg) override {
#ifndef __CUDA_ARCH__
        std::cout << "Sending message: count=" << count << ", type=" << type 
                  << ", dst=" << dst << ", tag=" << tag << std::endl;
#endif
        return CommonNetworkApi::sim_send(buffer, count, type, dst, tag, request, msg_handler, fun_arg);
    }

  private:
    static std::shared_ptr<Topology> topology;  ///< 拓扑实例
};

}  // namespace AstraSimAnalyticalCongestionUnaware
