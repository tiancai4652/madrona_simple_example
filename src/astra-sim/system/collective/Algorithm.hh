/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/topology/LogicalTopology.hh"

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSim {

/**
 * 集体通信算法的基类
 */
class Algorithm : public Callable {
  public:
    /// 算法类型枚举
    enum class Name { 
        Ring = 0,           ///< 环形算法
        DoubleBinaryTree,   ///< 双二叉树算法
        AllToAll,          ///< 全对全算法
        HalvingDoubling    ///< 对分倍增算法
    };

    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    Algorithm();
    
    /**
     * 虚析构函数
     */
    CUDA_HOST_DEVICE
    virtual ~Algorithm() = default;

    /**
     * 运行算法
     */
    CUDA_HOST_DEVICE
    virtual void run(EventType event, CallData* data) = 0;

    /**
     * 初始化算法
     */
    CUDA_HOST_DEVICE
    virtual void init(BaseStream* stream);

    /**
     * 处理事件回调
     */
    CUDA_HOST_DEVICE
    virtual void call(EventType event, CallData* data);

    /**
     * 退出算法
     */
    CUDA_HOST_DEVICE
    virtual void exit();

    Name name;              ///< 算法名称
    int id;                 ///< 算法ID
    BaseStream* stream;     ///< 关联的数据流
    LogicalTopology* logical_topo;  ///< 逻辑拓扑
    uint64_t data_size;    ///< 数据大小
    uint64_t final_data_size;  ///< 最终数据大小
    ComType comType;        ///< 通信类型
    bool enabled;           ///< 是否启用
};

}  // namespace AstraSim
