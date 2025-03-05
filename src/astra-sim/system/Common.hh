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

// 基本类型定义
using Tick = uint64_t;

constexpr uint64_t CLOCK_PERIOD = 1;           // 1ns
constexpr uint64_t FREQ = 1000 * 1000 * 1000;  // 1GHz

enum time_type_e { SE = 0, MS, US, NS, FS };

enum req_type_e { UINT8 = 0, BFLOAT16, FP32 };

struct timespec_t {
    time_type_e time_res;
    long double time_val;
};

// 事件类型
enum class EventType : uint8_t {
    General = 0,
    NetworkSend = 1,
    NetworkReceive = 2,
    MemoryAccess = 3,
    Compute = 4,
    Max
};

// 通信类型
enum class ComType : uint8_t {
    None = 0,
    Send = 1,
    Recv = 2,
    AllReduce = 3,
    AllGather = 4,
    ReduceScatter = 5,
    AllToAll = 6,
    Max
};

// 调度策略
enum class SchedulingPolicy : uint8_t {
    FIFO = 0,
    LIFO = 1,
    EXPLICIT = 2,
    Max
};

// 流状态
enum class StreamState : uint8_t {
    Created = 0,
    Ready = 1,
    Executing = 2,
    Finished = 3,
    Max
};

// 基本请求结构
struct alignas(8) sim_request {
    void* buffer;
    uint64_t size;
    int tag;
    bool active;
};

// 元数据基类
class MetaData {
public:
    timespec_t timestamp;
    CUDA_HOST_DEVICE virtual ~MetaData() = default;
};

// 调用数据基类
class CallData {
public:
    CUDA_HOST_DEVICE virtual ~CallData() = default;
};

enum class CollectiveOptimization { Baseline = 0, LocalBWAware };

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
    OneHalvingDoubling,
};

enum class CollectiveBarrier { Blocking = 0, Non_Blocking };

enum class IntraDimensionScheduling { FIFO = 0, RG, SmallestFirst, LessRemainingPhaseFirst };

enum class InterDimensionScheduling { Ascending = 0, OnlineGreedy, RoundRobin, OfflineGreedy, OfflineGreedyFlex };

enum class InjectionPolicy { Infinite = 0, Aggressive, SemiAggressive, ExtraAggressive, Normal };

enum class PacketRouting { Hardware = 0, Software };

enum class BusType { Both = 0, Shared, Mem };

class CloneInterface {
  public:
    virtual CloneInterface* clone() const = 0;
    virtual ~CloneInterface() = default;
};

class CollectiveImpl : public CloneInterface {
  public:
    CollectiveImpl(CollectiveImplType type) {
        this->type = type;
    };
    virtual CloneInterface* clone() const {
        return new CollectiveImpl(*this);
    }

    CollectiveImplType type;
};

class DirectCollectiveImpl : public CollectiveImpl {
  public:
    CloneInterface* clone() const {
        return new DirectCollectiveImpl(*this);
    };
    DirectCollectiveImpl(CollectiveImplType type, int direct_collective_window) : CollectiveImpl(type) {
        this->direct_collective_window = direct_collective_window;
    }

    int direct_collective_window;
};

}  // namespace AstraSim
