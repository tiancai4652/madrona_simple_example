#include "system_init.hpp"
#include "chakra_node_processing.hpp"
#include "topo_logic.hpp"
#include "sys_config.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

// Log formatting macros
#define LOG_SUBSEPARATOR "-------------------------------------------------------------------------------"

#if SYS_LOG
#define LOG_SYS_HEADER(sys_name, sys_desc) \
    printf("\n[SYS-%s] %s\n%s\n", sys_name, sys_desc, LOG_SUBSEPARATOR);
#define LOG_INFO(msg, ...) \
    printf("  INFO: " msg "\n", ##__VA_ARGS__);
#else
#define LOG_SYS_HEADER(sys_name, sys_desc) \
    do                                     \
    {                                      \
    } while (0)
#define LOG_INFO(msg, ...) \
    do                     \
    {                      \
    } while (0)
#endif

namespace madsimple::llm_system
{

    #define REGISTER_ARCHETYPE_WITH_NAME(registry, ArchetypeT) \
        registry.registerArchetype<ArchetypeT>()

// Macro for registering components with name printing  
#define REGISTER_COMPONENT_WITH_NAME(registry, ComponentT) \
    registry.registerComponent<ComponentT>()


    void registerTypes(ECSRegistry &registry)
    {
        registry.registerComponent<NpuID>();
        registry.registerComponent<ChakraNodes>();
        registry.registerComponent<ChakraTemplateNodes>();
        registry.registerComponent<InferenceNpuExecution>();
        registry.registerComponent<HardwareResource>();
        registry.registerComponent<ProcessingCompTask>();
        registry.registerComponent<ProcessingCommTasks>();
        registry.registerComponent<OneNPUFinishedFlag>();
        registry.registerComponent<ChakraNodesForNoDP>();
        registry.registerComponent<ChakraReadyState>();
        // 019 扇出保真: side 区清单 + INCAST 等齐进度.
        registry.registerComponent<ChakraFanouts>();
        registry.registerComponent<IncastWaitList>();
        registry.registerComponent<NpuFlowPairState>();
        // registry.registerArchetype<NpuNode>();
        REGISTER_ARCHETYPE_WITH_NAME(registry, NpuNode);

        registry.registerComponent<NextProcessTimes>();
        // registry.registerArchetype<NextProcessTimeE>();
        REGISTER_ARCHETYPE_WITH_NAME(registry, NextProcessTimeE);

        registry.registerComponent<CommModel>();
        // registry.registerArchetype<SysConfig>();
        REGISTER_ARCHETYPE_WITH_NAME(registry, SysConfig);

        registry.registerComponent<NodeID>();
        registry.registerComponent<TaskFlows>();
        // registry.registerArchetype<ProcessComm_E>();
        REGISTER_ARCHETYPE_WITH_NAME(registry, ProcessComm_E);

        // Register ring parameters component and archetype
        registry.registerComponent<RingParams>();
        // registry.registerArchetype<RingConfigEntity>();
        REGISTER_ARCHETYPE_WITH_NAME(registry, RingConfigEntity);

        // Register ring topology component and archetype
        registry.registerComponent<RingTopology>();
        // registry.registerArchetype<RingTopoEntity>();
        REGISTER_ARCHETYPE_WITH_NAME(registry, RingTopoEntity);

        registry.registerComponent<CheckNpuFinishFlag>();
        // registry.registerArchetype<CheckNpuFinishE>();
        REGISTER_ARCHETYPE_WITH_NAME(registry, CheckNpuFinishE);


        registry.registerComponent<RecvNodeFlag>(); 
        // registry.registerArchetype<ProcessRecvComm_E>();
        REGISTER_ARCHETYPE_WITH_NAME(registry, ProcessRecvComm_E);
    }

    void init(Engine &ctx)
    {
        ctx.data().ring_config_entity = Entity::none();
        ctx.data().sys_chakra_entities_created = false;
        // next process time entity
        Entity nextProcessTimeE = ctx.makeEntity<NextProcessTimeE>();
        for (size_t i = 0; i < NEXT_PROCESS_TIMES_CAP; i++)
        {
            ctx.get<NextProcessTimes>(nextProcessTimeE).times_abs[i] = 0;
        }
        ctx.data().next_process_time_entity = nextProcessTimeE;

        // // Sys_config
        // Entity sc = ctx.makeEntity<SysConfig>();
        // ctx.get<CommModel>(sc).all_reduce_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).all_gather_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).reduce_scatter_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).all_to_all_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).reduce_scatter_block_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).barrier_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).gather_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).scatter_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).broadcast_implementation = CommImplementationType::Ring;
        // ctx.get<CommModel>(sc).reduce_implementation = CommImplementationType::Ring;
        // ctx.data().sys_config_entity = sc;

//         // collective topo
//         RingTopology ring_topo(Dimension::Local, 2, 4, 1, 2);

// #if SYS_LOG
//         printf("ring_topo.id_to_index[0]: %d\n", ring_topo.id_to_index[0]);
//         printf("ring_topo.index_to_id[1]: %d\n", ring_topo.index_to_id[1]);
//         printf("ring_topo.index_to_id[2]: %d\n", ring_topo.index_to_id[2]);
//         printf("ring_topo.id_to_index[3]: %d\n", ring_topo.id_to_index[3]);
// #endif


        Entity e = ctx.makeEntity<CheckNpuFinishE>();
        ctx.get<CheckNpuFinishFlag>(e) = CheckNpuFinishFlag();
        ctx.data().check_npu_finish_entity = e;

        for (size_t i = 0; i < NPU_NUM; i++)
        {
            for (size_t j = 0; j < NPU_NUM; j++)
            {
                ctx.data().send_recv_map_recvend[i][j] = 0;
            }
        }

    }

    void sys_init(Engine &ctx, ChakraNodesData &chakra_nodes_data, ProcessParams &processParams)
    {
        if (processParams.params[998] == 0) {
            return;
        }
        if(ctx.data().sys_chakra_entities_created)
        {
            return;
        }
        
#if SYS_LOG
        printf("\n[SYS-0] Initialize NPUs\n%s\n", LOG_SUBSEPARATOR);
#endif
        // Get number of rows and columns
        size_t rows = static_cast<size_t>(processParams.params[100]);
        if (rows == 0 || rows > NPU_NUM) {
            ctx.singleton<SystemStatus>().failed = 1;
            ctx.singleton<SystemStatus>().error_code = 1;
            return;
        }
        size_t cols = sizeof(chakra_nodes_data.data[0]) / sizeof(chakra_nodes_data.data[0][0]); // Single row size / single element size
        size_t npu_nums = rows;
        size_t npu_data_num = cols;
#if SYS_LOG
        printf("  INFO: chakra_nodes_data bytes: total=%lu, per_row=%lu\n",
               (unsigned long)sizeof(chakra_nodes_data.data), (unsigned long)sizeof(chakra_nodes_data.data[0]));
        printf("  INFO: npu_count=%u, npu_data_len=%u\n", (unsigned int)npu_nums, (unsigned int)npu_data_num);
#endif

        for (size_t i = 0; i < npu_nums; i++)
        {
            Entity npuNode = ctx.makeEntity<NpuNode>();
            ctx.get<NpuID>(npuNode).value = i;
            ctx.get<NpuID>(npuNode).test_dim = false;
            ctx.get<NpuID>(npuNode).test_collective_comm_ring = false;
            int nodeCount = 0;
            parseChakraNodes(chakra_nodes_data, i, ctx.get<ChakraNodes>(npuNode).nodes, nodeCount);
            // B1: build the incremental ready set once at parse time.
            buildChakraReadyState(ctx.get<ChakraNodes>(npuNode),
                                   ctx.get<ChakraReadyState>(npuNode));
            // 019 扇出保真: 解码行尾 side 区.
            // E12: makeEntity 不跑构造, 先 zeroStruct (memset, 非展开),
            // 再填充; 不用逐条目聚合赋值 (1KB/条目 x 256 会炸 NVRTC).
            {
                auto &c = ctx.get<ChakraFanouts>(npuNode);
                zeroStruct(c);
                parseChakraFanouts(chakra_nodes_data, i,
                    ctx.get<ChakraNodes>(npuNode).nodes, nodeCount, c);
                // 强不变式: side 条目数必须等于主表扇出节点数 (防错位解析).
                {
                    int fanout_nodes = 0;
                    for (int32_t ni = 0; ni < nodeCount; ni++) {
                        ChakraNodeType t = ctx.get<ChakraNodes>(npuNode).nodes[ni].type;
                        if (t == ChakraNodeType::OUTCAST_NODE ||
                            t == ChakraNodeType::INCAST_NODE) {
                            fanout_nodes += 1;
                        }
                    }
                    if (fanout_nodes != static_cast<int>(c.num_entries)) {
                        ctx.singleton<SystemStatus>().failed = 1;
                        ctx.singleton<SystemStatus>().error_code = 1;
                        return;
                    }
                }
            }
            ctx.get<IncastWaitList>(npuNode) = IncastWaitList {};
            {
                auto &c = ctx.get<ChakraTemplateNodes>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_CHAKRA_NODES_PER_NPU; i++) {
                    c.nodes[i] = ChakraNode {};
                }
            }
            ctx.get<InferenceNpuExecution>(npuNode) = InferenceNpuExecution {};
            
            // Save nodeCount immediately
            const int savedNodeCount = nodeCount;
            if (savedNodeCount == 0)
            {
                ctx.destroyEntity(npuNode);
                // 打印：不是有效的node数据
                printf("  INFO: Not valid node data\n");
                return;
            }
            ctx.get<HardwareResource>(npuNode) = HardwareResource();
            ctx.get<ProcessingCompTask>(npuNode) = ProcessingCompTask();
            ctx.get<ProcessingCommTasks>(npuNode) = ProcessingCommTasks();
            ctx.get<ProcessingCommTasks>(npuNode).flow_id = i * FLOW_ID_MAX_LENGTH;
            ctx.get<OneNPUFinishedFlag>(npuNode) = OneNPUFinishedFlag();

#if SYS_LOG
            // Use %u instead of %zu to avoid GPU printf parameter misalignment bug
            printf("  INFO: NPU %u initialized with %d nodes\n", (unsigned int)i, savedNodeCount);
#endif
            ctx.data().npuEntities[i] = npuNode;
            ctx.data().numNpus = static_cast<int32_t>(i + 1);

            // A1/A7 draft components: makeEntity<NpuNode>() runs no
            // component constructors, so every component must be assigned
            // explicitly here or it starts as raw memory (this bit us once
            // -- garbage draft.count caused wild writes, see DECISIONS.md
            // 期2 E12; a second occurrence during C1 deleted the four
            // NpuFlow* assignments below -- garbage finished.count read as
            // 3.2e9 and crashed the A1 draft node, see E16).
            {
                auto &c = ctx.get<NpuFlowInbox>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_FLOWS_PER_NPU; i++) {
                    c.entries[i] = NpuFlowInboxEntry {};
                }
            }
            {
                auto &c = ctx.get<NpuFlowActiveList>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_FLOWS_PER_NPU; i++) {
                    c.active[i] = NpuActiveFlow {};
                }
            }
            {
                auto &c = ctx.get<NpuFlowFinishedList>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_FLOWS_PER_NPU; i++) {
                    c.flows[i] = SysFlowRecord {};
                }
            }
            {
                auto &c = ctx.get<NpuFlowPairState>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_NPU_FLOW_PAIR_STATES; i++) {
                    c.slots[i] = NpuFlowPairSlot {};
                }
            }
            {
                auto &c = ctx.get<NpuKvCollectDraft>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_FLOWS_PER_NPU; i++) {
                    c.tickets[i] = NpuKvCollectTicket {};
                }
            }
            {
                auto &c = ctx.get<NpuFlowScheduleDraft>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_FLOWS_PER_NPU; i++) {
                    c.tickets[i] = NpuScheduleTicket {};
                }
            }
            // 013-bug修复 M1: per-NPU flow-meta insert draft (same rule).
            {
                auto &c = ctx.get<NpuFlowCreateDraft>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_FLOWS_PER_NPU; i++) {
                    c.tickets[i] = NpuFlowCreateTicket {};
                }
            }
            // C1: per-NPU comm entity pool + retire draft (same E12 rule).
            ctx.get<NpuCommEntityPool>(npuNode) = NpuCommEntityPool {};
            {
                auto &c = ctx.get<NpuCommRetireDraft>(npuNode);
                zeroStruct(c);
#pragma unroll 1
                for (int32_t i = 0; i < MAX_COMM_ENTITIES_PER_NPU; i++) {
                    c.reqs[i] = NpuCommRetireReq {};
                }
            }
            NpuCommEntityPool &comm_pool = ctx.get<NpuCommEntityPool>(npuNode);
            for (uint32_t pool_idx = 0;
                 pool_idx < MAX_COMM_ENTITIES_PER_NPU; pool_idx++) {
                // ProcessComm_E pool entity. makeEntity runs no component
                // constructors (E12): every component is assigned here.
                Entity comm_entity = ctx.makeEntity<ProcessComm_E>();
                zeroStruct(ctx.get<NpuID>(comm_entity));
                zeroStruct(ctx.get<NodeID>(comm_entity));
                ctx.get<TaskFlows>(comm_entity) = TaskFlows {};
                zeroStruct(ctx.get<PoolSlotActive>(comm_entity));
                comm_pool.comm_free[pool_idx] = comm_entity;

                // ProcessRecvComm_E pool entity.
                Entity recv_entity = ctx.makeEntity<ProcessRecvComm_E>();
                zeroStruct(ctx.get<NpuID>(recv_entity));
                zeroStruct(ctx.get<NodeID>(recv_entity));
                zeroStruct(ctx.get<RecvNodeFlag>(recv_entity));
                zeroStruct(ctx.get<PoolSlotActive>(recv_entity));
                comm_pool.recv_free[pool_idx] = recv_entity;
            }
            comm_pool.comm_free_count = MAX_COMM_ENTITIES_PER_NPU;
            comm_pool.recv_free_count = MAX_COMM_ENTITIES_PER_NPU;
            NpuFlowPool &pool = ctx.get<NpuFlowPool>(npuNode);
            pool = NpuFlowPool {}; // small local, value-init kept
            pool.free_count = MAX_FLOWS_PER_NPU;
            for (uint32_t flow_idx = 0;
                 flow_idx < MAX_FLOWS_PER_NPU; flow_idx++) {
                Entity flow_entity = ctx.makeEntity<FlowMeta>();
                ctx.get<FlowDef>(flow_entity) = FlowDef {};
                ctx.get<FlowRouteState>(flow_entity) = FlowRouteState {};
                ctx.get<FlowRuntimeState>(flow_entity) = FlowRuntimeState {};
                ctx.get<FlowScheduleState>(flow_entity) = FlowScheduleState {};
                pool.free_flows[flow_idx] = flow_entity;
            }



// #if SYS_LOG_SPECIAL
//             printf("####  checke NPU %zu ####\n", i);
//             uint32_t independentNodes[100];
//             int count = checkRecvNodeDependencies(ctx.get<ChakraNodes>(npuNode), independentNodes, 100);

//             if (count == 1) {
//               
//             } else {
//                 printf("ERROR： %d Independent RECV NODE\n", count);
//                 for(int j=0; j<count; j++)
//                 {
//                     printf("  INFO: Independent RECV node %d\n", independentNodes[j]);
//                 }
//             }
// #endif

        }

 
        // Only destroy init_entity if it exists
        if (!ctx.data().sys_chakra_entities_created)
        {

            // ctx.destroyEntity(ctx.data().init_entity);
            ctx.data().sys_chakra_entities_created=true;
            SystemStatus &status = ctx.singleton<SystemStatus>();
            status.initialized = 1;
            status.total_npus = static_cast<int32_t>(npu_nums);
      

            // Create ring configuration entity and set ring parameters
            Entity ring_entity = ctx.makeEntity<RingConfigEntity>();
            ctx.get<RingParams>(ring_entity).ring_dim0 = static_cast<uint32_t>(processParams.params[0]);
            ctx.get<RingParams>(ring_entity).ring_dim1 = static_cast<uint32_t>(processParams.params[1]);
            ctx.get<RingParams>(ring_entity).ring_dim2 = static_cast<uint32_t>(processParams.params[2]);
            ctx.get<RingParams>(ring_entity).chunks_num = static_cast<uint32_t>(processParams.params[3]);
            ctx.get<RingParams>(ring_entity).npu_num = static_cast<uint32_t>(processParams.params[100]);
            ctx.data().ring_config_entity = ring_entity;



            // Create and initialize ring topology entities for three dimensions
            // Dimension 0: Local (offset=1)
            Entity ring_topo0_entity = ctx.makeEntity<RingTopoEntity>();
            ctx.get<RingTopology>(ring_topo0_entity).initialize(
                Dimension::Local, 
                ctx.get<RingParams>(ring_entity).ring_dim0, 
                1);
            ctx.data().ring_topo_entity[0] = ring_topo0_entity;

            // Dimension 1: Horizontal (offset=ring_dim0)
            Entity ring_topo1_entity = ctx.makeEntity<RingTopoEntity>();
            ctx.get<RingTopology>(ring_topo1_entity).initialize(
                Dimension::Horizontal, 
                ctx.get<RingParams>(ring_entity).ring_dim1, 
                ctx.get<RingParams>(ring_entity).ring_dim0);
            ctx.data().ring_topo_entity[1] = ring_topo1_entity;

            // Dimension 2: Vertical (offset=ring_dim0 * ring_dim1)
            Entity ring_topo2_entity = ctx.makeEntity<RingTopoEntity>();
            ctx.get<RingTopology>(ring_topo2_entity).initialize(
                Dimension::Vertical, 
                ctx.get<RingParams>(ring_entity).ring_dim2, 
                ctx.get<RingParams>(ring_entity).ring_dim0 * ctx.get<RingParams>(ring_entity).ring_dim1);
            ctx.data().ring_topo_entity[2] = ring_topo2_entity;


            // enum CollectiveCommType : uint64_t
            // {
            //     ALL_REDUCE = 0,
            //     REDUCE = 1,
            //     ALL_GATHER = 2,
            //     GATHER = 3,
            //     SCATTER = 4,
            //     BROADCAST = 5,
            //     ALL_TO_ALL = 6,
            //     REDUCE_SCATTER = 7,
            //     REDUCE_SCATTER_BLOCK = 8,
            //     BARRIER = 9
            // };
            // Sys_config
            Entity sc = ctx.makeEntity<SysConfig>();
            ctx.get<CommModel>(sc).all_reduce_implementation = (CommImplementationType)processParams.params[200];
            ctx.get<CommModel>(sc).reduce_implementation = (CommImplementationType)processParams.params[201];
            ctx.get<CommModel>(sc).all_gather_implementation = (CommImplementationType)processParams.params[202];
            ctx.get<CommModel>(sc).gather_implementation = (CommImplementationType)processParams.params[203];
            ctx.get<CommModel>(sc).scatter_implementation = (CommImplementationType)processParams.params[204];
            ctx.get<CommModel>(sc).broadcast_implementation = (CommImplementationType)processParams.params[205];
            ctx.get<CommModel>(sc).all_to_all_implementation = (CommImplementationType)processParams.params[206];
            ctx.get<CommModel>(sc).reduce_scatter_implementation = (CommImplementationType)processParams.params[207];
            ctx.get<CommModel>(sc).reduce_scatter_block_implementation = (CommImplementationType)processParams.params[208];
            ctx.get<CommModel>(sc).barrier_implementation = (CommImplementationType)processParams.params[209];
            
            
            ctx.data().sys_config_entity = sc;

            for(size_t i=0; i<NPU_NUM; i++)
            {
                for(size_t j=0; j<MAX_CHAKRA_NODES_PER_NPU; j++)
                {
                    ctx.data().npus_chakra_exec_entity[i][j] = Entity::none();

                }
            }



        }
        
    }

}
