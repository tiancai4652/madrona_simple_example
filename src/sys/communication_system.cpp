#include "communication_system.hpp"
#include "net_sys_interface.hpp"
#include "sys_config.hpp"
#include <cstdio>
#include <cstdlib>
#include "../sim.hpp"

using namespace madsimple;
using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

using madsimple::net_sys_interface::checkFlowFinish;
using madsimple::net_sys_interface::getCurrentTime;
using madsimple::net_sys_interface::setFlow;

// Log formatting macros
#define LOG_SUBSEPARATOR "-------------------------------------------------------------------------------"

#if SYS_LOG
#define LOG_SYS_HEADER(sys_name, sys_desc) \
    printf("\n[SYS-%s] %s\n%s\n", sys_name, sys_desc, LOG_SUBSEPARATOR);
#define LOG_TASK_START(task_name) \
    printf("  > %s\n", task_name);
#define LOG_TASK_SUCCESS(task_name) \
    printf("  [SUCCESS] %s\n", task_name);
#define LOG_NODE_COMPLETE(node_id) \
    printf("    Node [%d] completed\n", node_id);
#define LOG_INFO(msg, ...) \
    printf("  INFO: " msg "\n", ##__VA_ARGS__);
#else
#define LOG_SYS_HEADER(sys_name, sys_desc) \
    do                                     \
    {                                      \
    } while (0)
#define LOG_TASK_START(task_name) \
    do                            \
    {                             \
    } while (0)
#define LOG_TASK_SUCCESS(task_name) \
    do                              \
    {                               \
    } while (0)
#define LOG_NODE_COMPLETE(node_id) \
    do                             \
    {                              \
    } while (0)
#define LOG_INFO(msg, ...) \
    do                     \
    {                      \
    } while (0)
#endif

namespace madsimple::llm_system
{

    // Get the number of communications (ring) per phase
    int get_ring_comm_count_per_phase(CollectiveCommType comm_type,
                                      CommImplementationType comm_implementation_type, int nodes_in_ring)
    {
        (void)comm_implementation_type;
        int stream_count = 0;
        switch (comm_type)
        {
        case CollectiveCommType::ALL_REDUCE:
            stream_count = 2 * (nodes_in_ring - 1);
            break;
        case CollectiveCommType::ALL_TO_ALL:
            stream_count = ((nodes_in_ring - 1) * nodes_in_ring) / 2;
            break;
        default:
            stream_count = nodes_in_ring - 1;
        }
        return stream_count;
    }

    // Get the flow size per phase and the final data amount per node
    void get_ring_comm_size_per_flow(CollectiveCommType comm_type,
                                     CommImplementationType comm_implementation_type, uint64_t data_size, int nodes_in_ring, uint64_t &msg_size, uint64_t &final_data_size)
    {
        (void)comm_implementation_type;
        msg_size = data_size;
        switch (comm_type)
        {
        case CollectiveCommType::ALL_REDUCE:
            final_data_size = data_size;
            msg_size = data_size / nodes_in_ring;
            break;
        case CollectiveCommType::ALL_GATHER:
            final_data_size = data_size * nodes_in_ring;
            msg_size = data_size;
            break;
        case CollectiveCommType::REDUCE_SCATTER:
            final_data_size = data_size / nodes_in_ring;
            msg_size = data_size / nodes_in_ring;
            break;
        case CollectiveCommType::ALL_TO_ALL:
            final_data_size = data_size;
            msg_size = data_size / nodes_in_ring;
            break;
        default:
            msg_size = data_size;
            break;
        }
    }

    Direction get_comm_Ring_Direction()
    {
        //      if ((backend != AstraNetworkAPI::BackendType::Garnet || level > 0) && queues.size() > 1 &&
        //     allocator >= (queues.size() / 2)) {
        //     dir = RingTopology::Direction::Anticlockwise;
        // } else {
        //     dir = RingTopology::Direction::Clockwise;
        // }
        return Direction::Clockwise;
    }

    bool setNextExecFlows(Engine &ctx, TaskFlows &taskFlows, NpuID &npu_id)
    {
        // SysFlow flows_exec[MAX_FLOW_NUM_PER_COMM_NODE];
        uint32_t flow_exec_count = taskFlows.getNextExecFlows(npu_id.flows_exec);
#if SYS_LOG
        if (SYS_LOG_TARGET_NODE == npu_id.value)
        {
            printf("  INFO: setNextExecFlows: flow_exec_count=%u\n", flow_exec_count);
        }
#endif
        if (flow_exec_count > 0)
        {
            for (size_t i = 0; i < flow_exec_count; i++)
            {
                setFlow(ctx, npu_id.value, npu_id.flows_exec[i].comm_src, npu_id.flows_exec[i].comm_dst, npu_id.flows_exec[i].comm_size, npu_id.flows_exec[i].id, npu_id.flows_exec[i].comm_para);
            }
            npu_id.flows_exec_init();
            return true;
        }
        else
        {
            return false;
        }
    }

    // C1: append a retire request to the owner NPU's NpuCommRetireDraft.
    // Atomic slot claim (same pattern as net_sys_interface::addEvent) keeps
    // concurrent per-entity workers of the same NPU safe. The frame tail
    // recycleCommEntities singleton drains the draft in npu_id order and
    // moves the entities back to the pool free lists. Overflow is
    // structurally unreachable (retires per frame <= lent entities <=
    // MAX_COMM_ENTITIES_PER_NPU) and fails loudly with error_code 7.
    void retirePooledCommEntity(Engine &ctx, uint32_t npu_id_value,
                                Entity entity, uint32_t is_recv)
    {
        if (entity == Entity::none()) {
            return;
        }
        NpuCommRetireDraft &draft =
            ctx.get<NpuCommRetireDraft>(ctx.data().npuEntities[npu_id_value]);
        madrona::AtomicU32Ref count_ref(draft.count);
        uint32_t slot = count_ref.fetch_add_relaxed(1);
        if (slot >= MAX_COMM_ENTITIES_PER_NPU) {
            count_ref.fetch_sub<madrona::sync::relaxed>(1);
            draft.overflow_count += 1;
            SystemStatus &status = ctx.singleton<SystemStatus>();
            status.failed = 1;
            status.error_code = 7;
            return;
        }
        draft.reqs[slot] = NpuCommRetireReq {
            .entity = entity,
            .is_recv = is_recv,
        };
    }

    void sys_checkFlow(Engine &ctx, NpuID &npu_id, NodeID &node_id, TaskFlows &taskFlows)
    {
        #if SYS_LOG
        if (SYS_LOG_TARGET_NODE == npu_id.value)
        {
            LOG_SYS_HEADER("2", "Flow Status Check System");
            LOG_TASK_START("Checking network flow completion status");
        }
        #endif

        if (DEV_MODE == 0)
        {
            if (!taskFlows.is_exec)
            {
                taskFlows.is_exec = true;
                // first
                setNextExecFlows(ctx, taskFlows, npu_id);
            }
            else
            {
                taskFlows.setCurrentIndexFlowsFinish();

                // set next group flows
                if (!taskFlows.hasFlowWithCurrentIndex())
                {
                    if (!setNextExecFlows(ctx, taskFlows, npu_id))
                    {
                        if (taskFlows.areAllTasksDone())
                        {
                            #if SYS_LOG
                            if (SYS_LOG_TARGET_NODE == npu_id.value)
                            {
                                LOG_NODE_COMPLETE(node_id.value);
                            }
                            #endif

                            ctx.get<ProcessingCommTasks>(ctx.data().npuEntities[npu_id.value]).setFinish(node_id.value, getCurrentTime(ctx),npu_id.value);

                            // C1: the finished comm entity goes back to its
                            // owner NPU's pool via the retire draft (this
                            // worker runs per-entity in parallel, so a
                            // direct pool return would race with sibling
                            // entities of the same NPU -- F13). The frame
                            // tail recycleCommEntities step replays these
                            // requests in npu_id order.
                            retirePooledCommEntity(ctx, npu_id.value,
                                ctx.data().npus_chakra_exec_entity[npu_id.value][node_id.value], 0);
                        }
                    }
                }
            }
        }
        else
        {
            if (!taskFlows.is_exec)
            {
                taskFlows.is_exec = true;
                // first
                setNextExecFlows(ctx, taskFlows, npu_id);
            }
            else
            {
                // SysFlow flows_finish[MAX_FLOW_NUM_PER_COMM_NODE];
                // 019: 显式按缓冲大小限界 (npu_id.flows_finish 只有 128 槽;
                // 默认 512 会在同帧同 NPU 完成超 128 时踩内存. 限界后超量
                // 变可检测的未完成 hang, 不变静默 corruption).
                uint32_t flow_finish_count = checkFlowFinish(ctx, npu_id.value,
                    npu_id.flows_finish, MAX_FLOW_NUM_PER_COMM_NODE);

                if (flow_finish_count > 0)
                {
                    taskFlows.updateFlows(npu_id.flows_finish, flow_finish_count);

                    // set next group flows
                    if (!taskFlows.hasFlowWithCurrentIndex  ())
                    {
                        if (!setNextExecFlows(ctx, taskFlows, npu_id))
                        {
                            if (taskFlows.areAllTasksDone())
                            {
                                #if SYS_LOG
                                if (SYS_LOG_TARGET_NODE == npu_id.value)
                                {
                                    LOG_NODE_COMPLETE(node_id.value);
                                }
                                #endif

                                ctx.get<ProcessingCommTasks>(ctx.data().npuEntities[npu_id.value]).setFinish(node_id.value, getCurrentTime(ctx),npu_id.value);

                                // C1: retire via the owner NPU's draft (see
                                // the DEV_MODE==0 branch above).
                                retirePooledCommEntity(ctx, npu_id.value,
                                    ctx.data().npus_chakra_exec_entity[npu_id.value][node_id.value], 0);
                            }
                        }
                    }
                    npu_id.flows_finish_init();
                }
            }
        }
    }

    void sys_checkRecvFlow(Engine &ctx, NpuID &npu_id, NodeID &node_id, RecvNodeFlag &recvNodeFlag)
    {
        #if SYS_LOG
        if (SYS_LOG_TARGET_NODE == npu_id.value)
        {
            LOG_SYS_HEADER("3", "Flow Status Check recv flow System");
            LOG_TASK_START("Checking network flow completion status");
        }
        #endif

        if (DEV_MODE == 0)
        {
            if(ctx.data().send_recv_map_recvend[recvNodeFlag.comm_src][recvNodeFlag.comm_dst]==2)
            {
                #if SIMPLE_LOG_MODE
                if (SYS_LOG_TARGET_NODE == recvNodeFlag.comm_dst) {

                    printf("npus_chakra_exec_entity not none! npu_id[%d], node_id[%d]\n", npu_id.value, node_id.value);
                    printf("send_recv_map_recvend set 0, src(%lu),dst(%lu)\n", recvNodeFlag.comm_src, recvNodeFlag.comm_dst);
                }
                #endif
                ctx.data().send_recv_map_recvend[recvNodeFlag.comm_src][recvNodeFlag.comm_dst]=0;
                ctx.get<ProcessingCommTasks>(ctx.data().npuEntities[npu_id.value]).setFinish(node_id.value, getCurrentTime(ctx),npu_id.value);
                // C1: retire via the owner NPU's draft (parallel per-entity
                // workers must not touch the pool directly -- F13).
                retirePooledCommEntity(ctx, npu_id.value,
                    ctx.data().npus_chakra_exec_entity[npu_id.value][node_id.value], 1);
            }

        }
        else
        {
            bool matched = false;
            if (recvNodeFlag.flow_id != 0)
            {
                // Persistent per-flow pair state (keyed by comm_para): once
                // the matching SEND flow completes (state 2) this RECV can
                // finish regardless of when it fired. Unlike the single-step
                // NpuFlowFinishedList mailbox snapshot, this survives the
                // step boundary, so a RECV that fires after its flow already
                // completed still matches.
                matched = ctx.data().claimRecvDone(
                    ctx, recvNodeFlag.flow_id, recvNodeFlag.comm_src,
                    recvNodeFlag.comm_dst) != 0;
            }
            else
            {
                // comm_para == 0: fall back to the single-step mailbox
                // (src,dst) matching for workloads without flow_id.
                // 019: 同上, 按缓冲大小显式限界.
                uint32_t flow_finish_count = checkFlowFinish(
                    ctx, recvNodeFlag.comm_src,
                    npu_id.recv_node_flows_finish,
                    MAX_FLOW_NUM_PER_COMM_NODE);
                for (size_t i = 0; i < flow_finish_count; i++)
                {
                    if (npu_id.recv_node_flows_finish[i].comm_src == recvNodeFlag.comm_src &&
                        npu_id.recv_node_flows_finish[i].comm_dst == recvNodeFlag.comm_dst)
                    {
                        matched = true;
                        break;
                    }
                }
                npu_id.recv_node_flows_finish_init();
            }

            if (matched)
            {
                #if SIMPLE_LOG_MODE
                if (SYS_LOG_TARGET_NODE == recvNodeFlag.comm_dst) {
                    printf("recv flow done: npu_id[%d], node_id[%d]\n", npu_id.value, node_id.value);
                }
                #endif
                ctx.get<ProcessingCommTasks>(ctx.data().npuEntities[npu_id.value]).setFinish(node_id.value, getCurrentTime(ctx),npu_id.value);
                // C1: retire via the owner NPU's draft (see above).
                retirePooledCommEntity(ctx, npu_id.value,
                    ctx.data().npus_chakra_exec_entity[npu_id.value][node_id.value], 1);
            }
        }
    }

    void sys_checkIncastWaits(Engine &ctx, NpuID &npu_id,
        ProcessingCommTasks &comm_tasks, const ChakraFanouts &fanouts,
        IncastWaitList &waits)
    {
        const uint64_t dst = npu_id.value;
        for (uint32_t w = 0; w < waits.count;)
        {
            IncastWaitEntry &we = waits.waits[w];
            if (we.fanout_idx < 0 ||
                static_cast<uint32_t>(we.fanout_idx) >= fanouts.num_entries)
            {
                // 坏条目: 响亮失败 (error_code 3 = comm dst 家族).
                SystemStatus &status = ctx.singleton<SystemStatus>();
                status.failed = 1;
                status.error_code = 3;
                return;
            }
            const ChakraFanoutEntry &e = fanouts.entries[we.fanout_idx];
            for (uint32_t k = 0; k < e.count; ++k)
            {
                if (we.claimed[k >> 6] & (1ULL << (k & 63)))
                {
                    continue;
                }
                if (e.paras[k] != 0 &&
                    ctx.data().claimRecvDone(
                        ctx, e.paras[k], e.peers[k], dst) != 0)
                {
                    we.claimed[k >> 6] |= (1ULL << (k & 63));
                    we.done += 1;
                }
            }
            if (we.done >= e.count)
            {
                comm_tasks.setFinish(we.node_id, getCurrentTime(ctx), npu_id.value);
                waits.waits[w] = waits.waits[--waits.count];
            }
            else
            {
                ++w;
            }
        }
    }

}
