/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CollectivePhase.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/system/StreamStat.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/topology/LogicalTopology.hh"
#include "sys_layer/containers/FixedList.hpp"
#include "sys_layer/containers/FixedMap.hpp"

namespace AstraSim {

class RecvPacketEventHandlerData;

/**
 * 基础流类
 */
class BaseStream : public Callable, public StreamStat {
public:
    CUDA_HOST_DEVICE BaseStream(int stream_id, Sys* owner, custom::FixedList<CollectivePhase, 32> phases_to_go);
    CUDA_HOST_DEVICE virtual ~BaseStream() = default;

    CUDA_HOST_DEVICE void changeState(StreamState state);
    CUDA_HOST_DEVICE virtual void consume(RecvPacketEventHandlerData* message) = 0;
    CUDA_HOST_DEVICE virtual void init() = 0;

    static custom::FixedMap<int, int, 128> synchronizer;                    ///< 同步器映射
    static custom::FixedMap<int, int, 128> ready_counter;                   ///< 就绪计数器映射
    static custom::FixedMap<int, custom::FixedList<BaseStream*, 32>, 128> suspended_streams;  ///< 挂起流映射

    int stream_id;                                    ///< 流ID
    int total_packets_sent;                          ///< 已发送数据包总数
    SchedulingPolicy preferred_scheduling;            ///< 首选调度策略
    custom::FixedList<CollectivePhase, 32> phases_to_go;  ///< 待执行阶段列表
    int current_queue_id;                            ///< 当前队列ID
    CollectivePhase my_current_phase;                ///< 当前执行阶段
    ComType current_com_type;                        ///< 当前通信类型
    Tick creation_time;                              ///< 创建时间
    Tick last_init;                                  ///< 最后初始化时间
    Sys* owner;                                      ///< 所有者系统
    DataSet* dataset;                                ///< 数据集
    int steps_finished;                              ///< 已完成步骤数
    int initial_data_size;                           ///< 初始数据大小
    int priority;                                    ///< 优先级
    StreamState state;                               ///< 流状态
    bool initialized;                                ///< 是否已初始化

    Tick last_phase_change;                          ///< 最后阶段变更时间

    int test;                                        ///< 测试变量1
    int test2;                                       ///< 测试变量2
    uint64_t phase_latencies[10];                    ///< 阶段延迟数组
};

}  // namespace AstraSim
