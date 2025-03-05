/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdint>

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSim {

typedef unsigned long long Tick;

constexpr uint64_t CLOCK_PERIOD = 1;           // 1ns
constexpr uint64_t FREQ = 1000 * 1000 * 1000;  // 1GHz

// 时间类型枚举
enum class TimeType { 
    SE = 0, 
    MS, 
    US, 
    NS, 
    FS 
};

// 请求类型枚举
enum class ReqType { 
    UINT8 = 0, 
    BFLOAT16, 
    FP32 
};

// 时间规格结构
struct alignas(8) TimeSpec {
    TimeType time_res;
    long double time_val;
};

// 模拟请求结构
struct alignas(8) SimRequest {
    uint32_t srcRank;
    uint32_t dstRank;
    uint32_t tag;
    ReqType reqType;
    uint64_t reqCount;
    uint32_t vnet;
    uint32_t layerNum;
};

// 元数据基类
class MetaData {
public:
    CUDA_HOST_DEVICE virtual ~MetaData() = default;
    TimeSpec timestamp;
};

// 通信类型枚举
enum class ComType { 
    None = 0, 
    Reduce_Scatter, 
    All_Gather, 
    All_Reduce, 
    All_to_All, 
    All_Reduce_All_to_All 
};

// 集合通信优化枚举
enum class CollectiveOptimization { 
    Baseline = 0, 
    LocalBWAware 
};

// 集合通信实现类型枚举
enum class CollectiveImplType {
    Ring = 0,
    OneRing,
    Direct,
    OneDirect,
    AllToAll,
    DoubleBinaryTreeLocalAllToAll,
    LocalRingNodeA2AGlobalDBT,
    HierarchicalRing,
    DoubleBinaryTree,
    HalvingDoubling,
    OneHalvingDoubling
};

// 集合通信屏障枚举
enum class CollectiveBarrier { 
    Blocking = 0, 
    Non_Blocking 
};

// 调度策略枚举
enum class SchedulingPolicy { 
    LIFO = 0, 
    FIFO, 
    EXPLICIT, 
    None 
};

// 维度内调度枚举
enum class IntraDimensionScheduling { 
    FIFO = 0, 
    RG, 
    SmallestFirst, 
    LessRemainingPhaseFirst 
};

// 维度间调度枚举
enum class InterDimensionScheduling { 
    Ascending = 0, 
    OnlineGreedy, 
    RoundRobin, 
    OfflineGreedy, 
    OfflineGreedyFlex 
};

// 注入策略枚举
enum class InjectionPolicy { 
    Infinite = 0, 
    Aggressive, 
    SemiAggressive, 
    ExtraAggressive, 
    Normal 
};

// 数据包路由枚举
enum class PacketRouting { 
    Hardware = 0, 
    Software 
};

// 总线类型枚举
enum class BusType { 
    Both = 0, 
    Shared, 
    Mem 
};

// 流状态枚举
enum class StreamState { 
    Created = 0, 
    Transferring, 
    Ready, 
    Executing, 
    Zombie, 
    Dead 
};

// 事件类型枚举
enum class EventType {
    CallEvents = 0,
    General,
    RendezvousSend,
    RendezvousRecv,
    PacketReceived,
    PacketSent,
    Rec_Finished,
    Send_Finished,
    Processing_Finished,
    NPU_to_MA,
    MA_to_NPU,
    Consider_Process,
    Consider_Retire,
    Consider_Send_Back,
    StreamInit,
    CommProcessingFinished,
    CollectiveCommunicationFinished,
    CompFinished,
    MemLoadFinished,
    MemStoreFinished
};

// 克隆接口基类
class CloneInterface {
public:
    CUDA_HOST_DEVICE virtual CloneInterface* clone() const = 0;
    CUDA_HOST_DEVICE virtual ~CloneInterface() = default;
};

// 集合通信实现基类
class CollectiveImpl : public CloneInterface {
public:
    CUDA_HOST_DEVICE CollectiveImpl(CollectiveImplType type) {
        this->type = type;
    }
    
    CUDA_HOST_DEVICE virtual CloneInterface* clone() const override {
        return new CollectiveImpl(*this);
    }

    CollectiveImplType type;
};

// 直接集合通信实现类
class DirectCollectiveImpl : public CollectiveImpl {
public:
    CUDA_HOST_DEVICE CloneInterface* clone() const override {
        return new DirectCollectiveImpl(*this);
    }
    
    CUDA_HOST_DEVICE DirectCollectiveImpl(CollectiveImplType type, int direct_collective_window) 
        : CollectiveImpl(type) {
        this->direct_collective_window = direct_collective_window;
    }

    int direct_collective_window;
};

} // namespace AstraSim
