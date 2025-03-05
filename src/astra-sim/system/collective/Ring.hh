/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MyPacket.hh"
#include "astra-sim/system/collective/Algorithm.hh"
#include "astra-sim/system/topology/RingTopology.hh"
#include "sys_layer/containers/FixedList.hpp"

namespace AstraSim {

/**
 * 环形拓扑集体通信算法实现
 */
class Ring : public Algorithm {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    Ring(ComType type,
         int id,
         RingTopology* ring_topology,
         uint64_t data_size,
         RingTopology::Direction direction,
         InjectionPolicy injection_policy);

    /**
     * 运行算法
     */
    CUDA_HOST_DEVICE
    void run(EventType event, CallData* data) override;

    /**
     * 处理流计数
     */
    CUDA_HOST_DEVICE
    void process_stream_count();

    /**
     * 释放数据包
     */
    CUDA_HOST_DEVICE
    void release_packets();

    /**
     * 处理最大计数
     */
    CUDA_HOST_DEVICE
    virtual void process_max_count();

    /**
     * 减少计数
     */
    CUDA_HOST_DEVICE
    void reduce();

    /**
     * 检查是否可迭代
     */
    CUDA_HOST_DEVICE
    bool iteratable();

    /**
     * 获取非零延迟包的数量
     */
    CUDA_HOST_DEVICE
    virtual int get_non_zero_latency_packets();

    /**
     * 插入数据包
     */
    CUDA_HOST_DEVICE
    void insert_packet(Callable* sender);

    /**
     * 检查是否就绪
     */
    CUDA_HOST_DEVICE
    bool ready();

    /**
     * 退出算法
     */
    CUDA_HOST_DEVICE
    void exit();

    RingTopology::Direction dimension;  ///< 通信维度
    RingTopology::Direction direction;  ///< 通信方向
    MemBus::Transmition transmition;   ///< 传输类型
    int zero_latency_packets;          ///< 零延迟包数量
    int non_zero_latency_packets;      ///< 非零延迟包数量
    int id;                           ///< 节点ID
    int curr_receiver;                ///< 当前接收者
    int curr_sender;                  ///< 当前发送者
    int nodes_in_ring;                ///< 环中节点数
    int stream_count;                 ///< 流计数
    int max_count;                    ///< 最大计数
    int remained_packets_per_max_count;  ///< 每个最大计数剩余包数
    int remained_packets_per_message;    ///< 每个消息剩余包数
    int parallel_reduce;              ///< 并行规约数
    InjectionPolicy injection_policy;  ///< 注入策略
    custom::FixedList<MyPacket, 1024> packets;  ///< 数据包列表
    bool toggle;                      ///< 切换标志
    long free_packets;                ///< 空闲包数量
    long total_packets_sent;          ///< 已发送包总数
    long total_packets_received;       ///< 已接收包总数
    uint64_t msg_size;               ///< 消息大小
    custom::FixedList<MyPacket*, 1024> locked_packets;  ///< 锁定的数据包列表
    bool processed;                   ///< 是否已处理
    bool send_back;                   ///< 是否需要回发
    bool NPU_to_MA;                   ///< 是否从NPU到MA
};

}  // namespace AstraSim
