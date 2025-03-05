/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <vector>
#include <map>
#include <list>
#include <chrono>

#include "Common.hh"
#include "AstraNetworkAPI.hh"
#include "AstraRemoteMemoryAPI.hh"
#include "Callable.hh"
#include "CollectivePhase.hh"
#include "CommunicatorGroup.hh"
#include "MemBus.hh"
#include "Roofline.hh"
#include "UsageTracker.hh"
#include "topology/RingTopology.hh"
#include "../workload/Workload.hh"
#include "../../sys_layer/containers/FixedVector.hpp"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSim {

class BaseStream;
class StreamBaseline;
class DataSet;
class QueueLevels;
class Workload;
class LogicalTopology;
class BasicLogicalTopology;
class OfflineGreedy;

// 系统配置结构
struct alignas(8) SystemConfig {
    float simulationSpeed;
    float accuracy;
    int maxIterations;
    float peakPerformance;  // GFLOPS
    float localMemoryBandwidth;  // GB/s
};

class Sys : public Callable {
  public:
    // SchedulerUnit ------------------------------------------------------------
    class SchedulerUnit {
      public:
        CUDA_HOST_DEVICE
        SchedulerUnit(Sys* sys, std::vector<int> queues, int max_running_streams);
        CUDA_HOST_DEVICE
        void notify_stream_added(int vnet);
        CUDA_HOST_DEVICE
        void notify_stream_removed(int vnet, Tick running_time);

        Sys* sys;
        int max_running_streams;
        std::map<int, int> running_streams;
        custom::FixedVector<Tick, 32> latency_per_dimension;
    };
    //---------------------------------------------------------------------------

    // Constructor / Destructor -------------------------------------------------
    CUDA_HOST_DEVICE
    Sys(int id,
        const char* workload_configuration,
        const char* comm_group_configuration,
        const char* system_configuration,
        AstraRemoteMemoryAPI* remote_mem,
        AstraNetworkAPI* comm_NI,
        std::vector<int> physical_dims,
        std::vector<int> queues_per_dim,
        double injection_scale,
        double comm_scale,
        bool rendezvous_enabled);
    CUDA_HOST_DEVICE
    ~Sys();
    //---------------------------------------------------------------------------

    // Intialization ------------------------------------------------------------
    CUDA_HOST_DEVICE
    bool initialize_sys(const char* name);
    CollectiveImpl* generate_collective_impl_from_input(std::string collective_impl_str);
    //---------------------------------------------------------------------------

    // Helper Functions ---------------------------------------------------------
    static Tick boostedTick();
    static void sys_panic(std::string msg);
    //---------------------------------------------------------------------------

    // Simulation Loop ----------------------------------------------------------
    void exit_sim_loop(std::string msg);
    //---------------------------------------------------------------------------

    // General Event Handling ---------------------------------------------------
    CUDA_HOST_DEVICE
    void call(EventType type, CallData* data);
    CUDA_HOST_DEVICE
    void call_events();
    CUDA_HOST_DEVICE
    void register_event(Callable* callable, EventType event, CallData* callData, Tick delta_cycles);
    void try_register_event(Callable* callable, EventType event, CallData* callData, Tick& delta_cycles);
    static void handleEvent(void* arg);
    //---------------------------------------------------------------------------

    // Communicator Group Support -----------------------------------------------
    LogicalTopology* get_logical_topology(ComType comm_type);
    std::vector<CollectiveImpl*> get_collective_implementation(ComType comm_type);
    //---------------------------------------------------------------------------

    // Collective Communication Primitives --------------------------------------
    DataSet* generate_all_reduce(uint64_t size,
                                 std::vector<bool> involved_dimensions,
                                 CommunicatorGroup* communicator_group,
                                 int explicit_priority);
    DataSet* generate_all_to_all(uint64_t size,
                                 std::vector<bool> involved_dimensions,
                                 CommunicatorGroup* communicator_group,
                                 int explicit_priority);
    DataSet* generate_all_gather(uint64_t size,
                                 std::vector<bool> involved_dimensions,
                                 CommunicatorGroup* communicator_group,
                                 int explicit_priority);
    DataSet* generate_reduce_scatter(uint64_t size,
                                     std::vector<bool> involved_dimensions,
                                     CommunicatorGroup* communicator_group,
                                     int explicit_priority);
    DataSet* generate_collective(uint64_t size,
                                 LogicalTopology* topology,
                                 std::vector<CollectiveImpl*> implementation_per_dimension,
                                 std::vector<bool> dimensions_involved,
                                 ComType collective_type,
                                 int explicit_priority,
                                 CommunicatorGroup* communicator_group);
    CollectivePhase generate_collective_phase(ComType collective_type,
                                              BasicLogicalTopology* topology,
                                              uint64_t data_size,
                                              int queue_id,
                                              RingTopology::Direction direction,
                                              InjectionPolicy injection_policy,
                                              CollectiveImpl* collective_impl);
    int break_dimension(int model_parallel_npu_group);
    //---------------------------------------------------------------------------

    // Middle-level Network Primitives ------------------------------------------
    uint64_t determine_chunk_size(uint64_t& size, ComType type);
    int get_priority(int explicit_priority);
    void insert_into_ready_list(BaseStream* stream);
    void insert_stream(std::list<BaseStream*>* queue, BaseStream* baseStream);
    void ask_for_schedule(int max);
    void schedule(int num);
    void proceed_to_next_vnet_baseline(StreamBaseline* stream);
    //---------------------------------------------------------------------------

    // Low-level Network Primitives ---------------------------------------------
    CUDA_HOST_DEVICE
    int sim_send(Tick delay,
                 void* buffer,
                 uint64_t count,
                 int type,
                 int dst,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg);

    CUDA_HOST_DEVICE
    int sim_recv(Tick delay,
                 void* buffer,
                 uint64_t count,
                 int type,
                 int src,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg);

    int front_end_sim_send(Tick delay,
                           void* buffer,
                           uint64_t count,
                           int type,
                           int dst,
                           int tag,
                           sim_request* request,
                           void (*msg_handler)(void* fun_arg),
                           void* fun_arg);

    int front_end_sim_recv(Tick delay,
                           void* buffer,
                           uint64_t count,
                           int type,
                           int src,
                           int tag,
                           sim_request* request,
                           void (*msg_handler)(void* fun_arg),
                           void* fun_arg);

    int rendezvous_sim_send(Tick delay,
                            void* buffer,
                            uint64_t count,
                            int type,
                            int dst,
                            int tag,
                            sim_request* request,
                            void (*msg_handler)(void* fun_arg),
                            void* fun_arg);

    int rendezvous_sim_recv(Tick delay,
                            void* buffer,
                            uint64_t count,
                            int type,
                            int src,
                            int tag,
                            sim_request* request,
                            void (*msg_handler)(void* fun_arg),
                            void* fun_arg);
    //---------------------------------------------------------------------------

    static std::vector<Sys*> all_sys;  // vector of all Sys objects

    int id;
    bool initialized;

    // workload
    Workload* workload;

    // roofline model
    bool roofline_enabled;
    double peak_perf;
    Roofline* roofline;

    // memory
    double local_mem_bw;
    AstraRemoteMemoryAPI* remote_mem;

    // memory bus
    MemBus* memBus;
    float inp_L;
    float inp_o;
    float inp_g;
    float inp_G;
    bool model_shared_bus;
    double injection_scale;
    int communication_delay;
    int local_reduction_delay;

    // network
    AstraNetworkAPI* comm_NI;
    double comm_scale;
    bool rendezvous_enabled;

    // scheduler
    SchedulerUnit* scheduler_unit;
    QueueLevels* vLevels;
    OfflineGreedy* offline_greedy;
    IntraDimensionScheduling intra_dimension_scheduling;
    InterDimensionScheduling inter_dimension_scheduling;
    int round_robin_inter_dimension_scheduler;
    int active_chunks_per_dimension;
    int priority_counter;
    uint64_t pending_events;
    int preferred_dataset_splits;
    int concurrent_streams;
    int active_first_phase;
    int max_running;

    // for supporting LIFO
    std::list<BaseStream*> ready_list;
    SchedulingPolicy scheduling_policy;
    int first_phase_streams;
    int total_running_streams;
    std::map<int, std::list<BaseStream*>> active_Streams;
    std::map<int, std::list<int>> stream_priorities;

    custom::FixedVector<std::tuple<Callable*, EventType, CallData*>, 32> event_queue;
    int total_nodes;
    int dim_to_break;
    std::vector<int> logical_broken_dims;

    std::vector<int> physical_dims;
    std::vector<int> queues_per_dim;

    // collective communication
    int num_streams;
    static uint8_t* dummy_data;
    std::map<std::string, LogicalTopology*> logical_topologies;
    std::vector<CollectiveImpl*> all_reduce_implementation_per_dimension;
    std::vector<CollectiveImpl*> reduce_scatter_implementation_per_dimension;
    std::vector<CollectiveImpl*> all_gather_implementation_per_dimension;
    std::vector<CollectiveImpl*> all_to_all_implementation_per_dimension;
    CollectiveOptimization collectiveOptimization;
    Tick last_scheduled_collective;
    bool break_dimension_done;
    int dimension_to_break;

    // statistics
    bool trace_enabled;

    // skip simulation for all nodes and use current duration
    bool replay_only;

    // 性能模型
    SystemConfig config;
};

}  // namespace AstraSim
