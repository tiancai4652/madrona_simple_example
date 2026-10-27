#pragma once

#include "../sim.hpp"
#include "../types.hpp"
#include "topo_logic.hpp"

namespace madsimple::llm_system
{
    // Import madrona types into current namespace
    using madrona::ECSRegistry;

    // Communication flow management functions
    bool setNextExecFlows(Engine &ctx, TaskFlows &taskFlows, NpuID &npu_id);
    void sys_checkFlow(Engine &ctx, NpuID &npu_id, NodeID &node_id, TaskFlows &taskFlows);
    void sys_checkRecvFlow(Engine &ctx, NpuID &npu_id, NodeID &node_id, RecvNodeFlag &recvNodeFlag);
    // 019 扇出保真: INCAST 等齐轮询 (逐帧 claimRecvDone, 齐 127 条才 finish).
    void sys_checkIncastWaits(Engine &ctx, NpuID &npu_id,
        ProcessingCommTasks &comm_tasks, const ChakraFanouts &fanouts,
        IncastWaitList &waits);

    // Ring communication helper functions
    int get_ring_comm_count_per_phase(CollectiveCommType comm_type,
                                      CommImplementationType comm_implementation_type,
                                      int nodes_in_ring);

    void get_ring_comm_size_per_flow(CollectiveCommType comm_type,
                                     CommImplementationType comm_implementation_type,
                                     uint64_t data_size, int nodes_in_ring,
                                     uint64_t &msg_size, uint64_t &final_data_size);

    Direction get_comm_Ring_Direction();
}
