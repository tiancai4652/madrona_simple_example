#pragma once

#include <cstdint>

#include <madrona/components.hpp>
#include <madrona/sync.hpp>

namespace madsimple {

// GPU compile-time safety valve: NVRTC -O3 fully unrolls wholesale
// `x = T{}` assignments of multi-KB components into tens of thousands of
// individual stores, making every TU that runs them pay minutes of compile
// time (PortBuffer ~64KB, InferenceRuntime ~300KB, constant-bound clear
// loops, etc; see /app/report/优化gpu编译速度/8*.md). zeroBytes keeps the
// stores in a non-unrollable loop (runtime ~ memset speed: word-wise with
// pragma unroll 1).
//
// SEMANTIC CONTRACT: only use on structs whose value-init `T{}` is all-zero
// bytes. Structs with non-zero NSDMIs (e.g. `= -1` or `= Entity::none()`,
// which is 0xFFFF'FFFF) must NOT go through this: byte-zeroing would change
// their initial state. Each call site must be audited (audited 2026-09-22,
// see report 8 for the per-struct NSDMI check).
inline MADRONA_NO_INLINE void zeroBytes(void *ptr, uint32_t num_bytes)
{
    char *p = static_cast<char *>(ptr);
    while (((reinterpret_cast<uintptr_t>(p) & 7u) != 0) && num_bytes > 0) {
        *p++ = 0;
        num_bytes -= 1;
    }
    uint64_t *word = reinterpret_cast<uint64_t *>(p);
    uint32_t num_words = num_bytes / 8;
#pragma unroll 1
    for (uint32_t i = 0; i < num_words; i++) {
        word[i] = 0;
    }
    p += num_words * 8;
    uint32_t num_tail = num_bytes % 8;
#pragma unroll 1
    for (uint32_t i = 0; i < num_tail; i++) {
        p[i] = 0;
    }
}

template <typename T>
inline MADRONA_NO_INLINE void zeroStruct(T &obj)
{
    zeroBytes(&obj, static_cast<uint32_t>(sizeof(T)));
}

enum class ExportID : uint32_t {
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    // GPU-mode introspection mirrors: a final per-step system copies the
    // scalar counts / simulation time and the flowCompletions[] array
    // out of the Sim struct into these singleton components on
    // SimDriverArch so mgr.cpp's GPUImpl can cudaMemcpy them back to
    // host. On CPU these are unused (CPUImpl reads Sim fields directly
    // via TaskGraphExecutor::getWorldData), but we still populate them
    // so the export layout stays identical between backends.
    SimStats,
    FlowCompletionBuf,
    StepPhaseTimes,
    ChakraNodesData,
    ProcessParams,
    SystemStatus,
    InferenceConfigData,
    InferenceRequestData,
    WorkloadParamsTableData,
    InferenceStatsData,
    NumExports,
};

struct Reset {
    int32_t resetNow;
};

enum class Action : int32_t {
    Up    = 0,
    Down  = 1,
    Left  = 2,
    Right = 3,
    None,
};

struct GridPos {
    int32_t y;
    int32_t x;
};

struct Reward {
    float r;
};

struct Done {
    float episodeDone;
};

struct CurStep {
    uint32_t step;
};

struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep
> {};

// Singleton driver used by task-graph-level systems that need a single
// parallel entry point (scheduling, event delivery, global reductions,
// singleton flush nodes). per-entity parallel systems must not live on
// SimDriverArch; they should iterate over Port / FlowTag directly.
struct SimDriver {
    int32_t tick = 0;
};

using Time = double;
using Bw = double;
using Bytes = double;
using FlowId = int64_t;
using NodeId = int32_t;

// =============================================================================
// Capacity constants
// =============================================================================
//
// Keep these together so scale changes such as d64 -> d256 can be adjusted in
// one place. Topology / flow-global limits live in sim.hpp; these constants
// cover exported buffers and per-port local queues/components declared below.
constexpr int32_t MAX_FLOW_COMPLETIONS = 1048576; // 018-layer0-GPUfit16: 4M->1M（省144MB；export超量截断，sim已解耦不受影响）。
// full1024 layer0 serving总completion超677888（export满后旧代码early-return跳过markSendDone等仿真记账，致D侧干净死锁）。
// 记账已与export解耦（sim_systems.cpp），此处只定export镜像容量（单条48B，4M=192MB，可承受）。
// 若未来总completion超4M，export按sim_tasks钳位截断但仿真不受影响（fail-soft）。
constexpr int32_t PFC_MAX_PRIORITY = 1;

// Must be >= the maximum number of flows/events per hot port so buffer chunks
// and per-port queues do not silently drop same-timestamp fan-in.
constexpr int32_t MAX_CHUNK_WEIGHTS = 512; // 018-layer0-GPUfit27: 1024->512（给chunks翻倍腾显存；gpu64全程weights pmax<=394，余量30%；若FATAL响亮回退）。
// burst下首chunk backlog超512（incast 127×KV多层叠加；旧FATAL打印值被并发冲花，
// 但outbox同款已证“raise即过=真burst”）。BufferChunk 8KB->32KB（+2.3GB），总显存~40GB仍fit。
// 超限响亮FATAL，绝不静默。
// 018-layer0-GPUfit: inst128实测chunks零使用（HWM未触发），KV burst未知；128→32
// 降单端口PortBuffer 8MB→2MB（GPU显存关键路径）。超限由带宽侧FATAL响亮报错，
// 绝不静默覆盖（见 materializeAllocPortBuffer）。
constexpr int32_t MAX_BUFFER_CHUNKS = 128; // 021-fit28: 256->128（链化后全程pmax<=26，5x余量；省PortBuffer约3.2GB/2.1MB->1.05MB每站）。超限响亮FATAL。
constexpr int32_t MAX_PAUSED_UPSTREAMS = 64;

// Per-port cleanup / completion / event queues. These are local fixed-size
// batches, so they must cover the largest same-port fan-in in one step.
// Restored to the leafspine1024-d128 tuned values (62232f1 shrank them for a
// different experiment, which caused silent arrival-event loss and ghost
// flows here; the B1 iteration briefly jumped them to 4096, but deep queues
// changed the engine's event coalescing regime -- livelock + non-symmetric
// per-flow completion timing, see CHECKPOINT-REPORT §19 ablation matrix).
// pushDelayedEvent now FAILS loudly on overflow (error_code=6), so an
// underestimated capacity in a future workload surfaces as a failed run
// rather than wrong results. Per-experiment tuning stays compile-time.
constexpr int32_t MAX_PORT_CLEANUP = 1024;
constexpr int32_t MAX_PORT_OUTBOX = 8192; // 018-layer0-GPUfit20: 4096->8192（16384装不下init OOM；8192大概率够用，不够再FATAL加）。
// burst（896 NPU同批完成）下热端口单步Arrival/BwUpdate事件超2048（step911三worker同爆）。
// +590KB/端口（196KB->786KB ×3072=+1.8GB device）。超限响亮FATAL，绝不静默。
constexpr int32_t MAX_PORT_TAG_LOOKUP = 8192; // 021-fit28: 与 MAX_TAGS_PER_PORT 同步回落 (掩码2的幂，8192合法；省0.77GB)。
constexpr int32_t MAX_PORT_DELAYED_EVENTS = 2048; // 021-fit28: 4096->2048（链化后全程pmax<=252，8x余量；省0.8GB；超限静默丢但error 6熔断响亮）。
// 018-layer0: layer0 full1024 死于 step387 (sim 348ms)：单端口并发流超 1024（incast 多波次 + KV 128 流叠加），
// createTagOnPort 报 capacity exceeded MAX_TAGS_PER_PORT=1024 on port 256。1024→4096。
// 018-layer0-GPUfit9: D-finish burst全局14万active tags（digest step1200），热端口超4096（step1041/2124两爆）。
// 4096->8192（tag Entity数组7×32KB×3072=+350MB，scratch tags/live +100MB，lookup +200MB，总+~0.7GB）。
// 超限响亮FATAL，绝不静默。
constexpr int32_t MAX_TAGS_PER_PORT = 8192; // 021-fit28: 16384->8192（链化后全程pmax<=198，40x余量；5858那次是无链版不适用；省Tag系约2.0GB，FrameScratch点名册同步减半）。超限响亮FATAL。
// Do not create FlowTag entities inside the single-threaded GPU initWorlds
// kernel. Tags are allocated lazily by createTagOnPort and recycled through
// PortTagPool after first use.
constexpr int32_t INITIAL_TAGS_PER_PORT = 0;
constexpr int32_t MAX_TAGS_PER_INGRESS = 8192; // 018-layer0-B: retry1 死于 step388 flushIngressTagLinks IngressTagList overflow (1024)；同族三墙打包升 4096
// 018-layer0-GPUfit10: D-finish burst再爆 (step2126, 单FATAL干净打印)。4096->8192（+100MB，可忽略；DIRTY_MARKS派生自动跟随）。
constexpr int32_t MAX_PORT_INBOX_ARRIVAL = 4096; // 018-layer0-B: 同族第二墙 (incast+KV 叠加 arrival burst)，2048→4096；DELAYED_EVENTS 保持 8192 不动
constexpr int32_t MAX_PORT_INBOX_BWUPD = 2048;
constexpr int32_t MAX_PORT_INBOX_PFC = 512;
constexpr int32_t MAX_PORT_CREATE = 4096; // 018-layer0-B: 同族第三墙 (arrival/bwupdate tag create)，2048→4096；MAX_PORT_INGRESS_LINKS 派生自动跟随
constexpr int32_t MAX_PORT_INGRESS_LINKS = MAX_PORT_CREATE;
constexpr int32_t MAX_PORT_DIRTY_MARKS = MAX_TAGS_PER_INGRESS; // 派生宏自动跟随 INGRESS→4096，无需手动改 (已确认)
constexpr int32_t MAX_PORT_COMPLETE = 1024;
constexpr int32_t MAX_PORT_INGRESS_UNLINKS = MAX_PORT_CLEANUP;
// Time-skip event queue (SystemEventQueue) capacity. Raised from 1024 so the
// sys-layer COMP/COMM scheduling events can't silently fill the queue and be
// dropped for realistic workloads; skipTime_remove_time now also reclaims
// consumed slots so the count stays bounded by the number of active events.
constexpr uint32_t MAX_SYSTEM_EVENTS = 16384;
constexpr uint32_t MAX_FAKE_FINISHED_FLOWS = 16;

// Per-NPU flow bookkeeping capacity constants. Unlike the old design
// (single global PendingSetFlowQueue + a monotonically-growing
// flowMetaEntities[] array), each NPU owns a small *bounded* and *reused*
// pool of FlowMeta entities plus its own inbox/active/finished lists. This
// caps total dynamic-flow memory at MAX_NPUS * MAX_FLOWS_PER_NPU instead of
// "total flows ever created over the whole run", and lets every NPU's
// setFlow()/materialize/schedule/checkFlowFinish work purely on its own
// entity's components with zero cross-NPU locking. See
// report/2-merge-plan/实施/阶段A2-per-NPU流并行化方案.md for the full design.
constexpr uint32_t MAX_NPUS = 1024;
// 018-layer0-GPUfit: layer0 chakra-only 每 NPU 峰值 254 流
// (2 OUTCAST x 127 同拍 burst; inbox/active/finished/draft 同界).
// 018-layer0-GPUfit6: 2048->8192。解耦修完D侧推进后，真峰值并发>2048
// （61层×128shard KV波次叠加 + fanout 254; step910 pool耗尽实测）。
// 8192覆盖~8000上界（4x余量）；16384原值device装不下（16M FlowMeta+14GB池）。
// 内存代价约+7GB（FlowMeta 2M->8M + 各per-NPU表×4），GPU峰值~35GB仍fit。
// 溢出响亮报 error_code 4 / FATAL（createFlows），绝不静默。
// 本宏同时界定 NpuFlowInbox/Active/Finished/KvCollect/Schedule/CreateDraft +
// NpuFlowPool 预分配 FlowMeta 实体数. 溢出响亮报 error_code 4 / FATAL.
// 注意: 019 全扇出 (15494 瞬时) 在本构建下会响亮溢出, 需回退重编.
constexpr uint32_t MAX_FLOWS_PER_NPU = 2048; // 018-layer0-GPUfit25: 8192->2048（分流后每NPU KV~128+data254，2048=4x余量；省~5GB。若FATAL则回调4096）。

// Because NPU-owned FlowMeta entities are recycled through NpuFlowPool, the
// per-entity completion_record does not survive slot reuse. This bounded
// append-only log keeps their FCT records durable so
// num_flow_completions()/flow_completion(idx) stay consistent for
// dynamically injected flows (the static flow-file path keeps rebuilding
// its records straight from flowMetaEntities[] as before).
constexpr int32_t MAX_DYNAMIC_FLOW_COMPLETIONS = 16384;

// Matches the layout of the completion record snapshotted by
// Sim::recordFlowCompletion(). Kept POD so it can be copied into the
// FlowCompletionBuf component that GPUImpl DMAs back to host every
// step.
struct FlowCompletionRecord {
    FlowId flow_id = -1;
    NodeId src_node = -1;
    NodeId dst_node = -1;
    Bytes size = 0.0;
    Time start_time = 0.0;
    Time end_time = 0.0;
    int32_t priority = 0;
    int32_t _pad = 0;

    inline Time fct() const
    {
        return end_time - start_time;
    }
};

struct SysFlowRecord {
    uint32_t owner_npu_id = 0;
    uint32_t flow_id = 0;
    uint64_t comm_src = 0;
    uint64_t comm_dst = 0;
    uint64_t flow_size = 0;
    uint64_t comm_para = 0;
    uint64_t start_time_ns = 0;
    uint64_t end_time_ns = 0;
};

struct SystemEventQueue {
    uint64_t times_ns[MAX_SYSTEM_EVENTS] {};
    uint32_t count = 0;
    uint32_t overflow_count = 0;
};

// ---------------------------------------------------------------------
// Per-NPU flow bookkeeping (replaces the old global PendingSetFlowQueue /
// CompletedSysFlowSnapshot / flowMetaEntities[]-based scheduling window
// for dynamically injected (setFlow) flows).
//
// Design invariant (see design doc for the full argument): net_sys_
// interface::setFlow(ctx, npu_id, ...) is only ever called by NPU
// `npu_id`'s *own* step-system invocation (i.e. a per-NPU parallel worker
// submitting a flow it itself owns). Under that invariant, writing into
// this NPU's own NpuFlowInbox from inside that worker touches only this
// entity's own column storage, so no lock is required even when many
// NPUs call setFlow concurrently in the same ParallelForNode.
// ---------------------------------------------------------------------

struct NpuFlowInboxEntry {
    uint64_t comm_src = 0;
    uint64_t comm_dst = 0;
    uint64_t flow_size = 0;
    uint32_t flow_id = 0;
    uint64_t comm_para = 0;
};

// Lock-free per-NPU inbox for setFlow() requests. Drained once per step by
// the (serial, but O(active NPUs) not O(total flows ever))
// createFlowsFromNpuRequests singleton.
struct NpuFlowInbox {
    uint32_t count = 0;
    uint32_t overflow_count = 0;
    NpuFlowInboxEntry entries[MAX_FLOWS_PER_NPU] {};
};

// Preallocated FlowMeta entity pool owned by this NPU, mirroring the
// PortTagPool pattern used for FlowTag entities: all MAX_FLOWS_PER_NPU
// FlowMeta entities are created once when the NPU entity is initialized,
// and recycled through this free list. No ctx.makeEntity/destroyEntity
// call is ever needed on the per-step hot path.
struct NpuFlowPool {
    uint32_t free_count = 0;
    madrona::Entity free_flows[MAX_FLOWS_PER_NPU] {};
};

struct NpuActiveFlow {
    madrona::Entity flow_entity = madrona::Entity::none();
    uint32_t flow_id = 0;
};

// Flows this NPU currently has materialized and in flight (pending or
// active, not yet completed). Bounded to this NPU alone, so a still-
// pending flow belonging to NPU A (or the static flow-file batch) can
// never block NPU B's ready flow from being scheduled -- this is the
// head-of-line-blocking fix relative to the old single global
// pendingFlowCursor window shared by every flow source.
struct NpuFlowActiveList {
    uint32_t count = 0;
    NpuActiveFlow active[MAX_FLOWS_PER_NPU] {};
};

// Completion mailbox for this NPU. Appended to by the network layer's
// (already serial) recordFlowCompletion() whenever a flow whose
// owner_npu_id matches this NPU finishes; read via checkFlowFinish() and
// cleared via clearFlowFinishQueue(), both scoped to this NPU only so an
// O(all NPUs' completions this step) scan is never required.
struct NpuFlowFinishedList {
    uint32_t count = 0;
    uint32_t overflow_count = 0;
    SysFlowRecord flows[MAX_FLOWS_PER_NPU] {};
};

// --- A1 (KV collect) draft pattern -----------------------------------------
// Parallel per-NPU draft: each NPU scans its own NpuFlowFinishedList and
// records raw KV-completion candidates (slot, shard, end_time) WITHOUT
// touching the shared InferenceRuntime. A serial commit step then replays
// every NPU's tickets in npu_id ascending order (== the legacy singleton's
// iteration order), so the ledger mutations (kv_done_mask OR, t_kv_done
// max, kv_done count) and -- critically -- the D-worker queue enqueue
// order stay byte-identical. Capacity matches the finished list bound, so
// overflow is structurally unreachable; it still fails loudly per the
// accepted design.
struct NpuKvCollectTicket {
    uint32_t slot = 0;
    uint32_t shard = 0;
    uint64_t end_time_ns = 0;
};

struct NpuKvCollectDraft {
    uint32_t count = 0;
    uint32_t overflow_count = 0;
    NpuKvCollectTicket tickets[MAX_FLOWS_PER_NPU] {};
};

// Persistent SEND completion mailbox owned by one destination NPU.
// SEND completion is published from the serial network flush; concurrent
// RECV systems only scan and atomically consume matching ready entries.
struct NpuFlowPairSlot {
    uint64_t comm_para = 0;
    uint64_t comm_src = 0;
    uint64_t comm_dst = 0;
    int32_t state = 0; // 0 = reusable, 2 = SEND completed
};

// 018-layer0-GPUfit: layer0 每 NPU 2 OUTCAST x 127 + 2 INCAST x 127 = 最坏堆积
// 约 508 个 (para,src,dst) 槽位 (SEND 先 mark, INCAST claim 后复用; KV 不走 pair).
// 取 1024 留 2x 余量 (1024x32B = 32KB/NPU, 原 512KB/NPU). 溢出响亮报 error_code 5.
// 注意: 019 全扇出 (15494) 在本构建下会响亮溢出, 需回退重编 (见 018 收敛清单).
constexpr uint32_t MAX_NPU_FLOW_PAIR_STATES = 1024; // 018-layer0-GPUfit12: 16384->1024 (需508, 2x余量; 省480MB对冲pool上涨)。溢出响亮error 5。

struct NpuFlowPairState {
    uint32_t count = 0;
    uint32_t overflow_count = 0;
    NpuFlowPairSlot slots[MAX_NPU_FLOW_PAIR_STATES] {};
};

// --- C1 (comm task entity pool) --------------------------------------------
// Preallocated per-NPU pools of ProcessComm_E / ProcessRecvComm_E entities,
// removing ctx.makeEntity/destroyEntity traffic from the per-step hot path
// (the legacy dynamic creation is a global-allocator hotspot on GPU).
//
// Lifecycle: sys_init preallocates MAX_COMM_ENTITIES_PER_NPU entities of
// each kind per NPU and fills the free lists. The per-NPU process*Node
// systems take entities from their OWN pool (serial within the NPU
// instance); finished comm/recv entities mark themselves retired by
// atomically appending a retire request to the owner NPU's
// NpuCommRetireDraft (per-entity writers, atomic slot claim -- same pattern
// as net_sys_interface::addEvent), and a light singleton at the frame tail
// (recycleCommEntities) replays the requests in npu_id order and moves the
// entities back to the free lists. A same-NPU retire-from-decision-path
// (recv immediate-finish, coll re-issue) goes straight back to the pool:
// it runs serially inside that NPU's own process*ChakraNodes instance.
// Pool exhaustion fails loudly with error_code 7.
// 018-layer0-GPUfit4: layer0 每 NPU 并发 comm/recv 实体峰值各<=4
// (2 OUTCAST + 2 INCAST; ProcessComm_E/Recv 分池; KV 不占池, 已核查 kv_transfer/inference_runtime 无 takeComm).
// 取 32 留 8x 余量 (原 256, 8x 内存削减; ProcessComm_E 38KB/行, 主因 NpuID 37KB).
// 实体 262144->32768: NpuID 列 9.9GB->1.2GB + TaskFlows 列 1.9GB->0.24GB + 排序暂存 8GB->1GB.
// 耗尽响亮报 error_code 7 (llm_system.cpp 三处 take + retire 路径), 绝不静默.
constexpr uint32_t MAX_COMM_ENTITIES_PER_NPU = 8; // 018-layer0-GPUfit16: 16->8（layer0并发<=4，仍2x余量；省0.5GB）。耗尽响亮error 7。

struct NpuCommEntityPool {
    madrona::Entity comm_free[MAX_COMM_ENTITIES_PER_NPU] {};
    int32_t comm_free_count = 0;
    madrona::Entity comm_lent[MAX_COMM_ENTITIES_PER_NPU] {};
    int32_t comm_lent_count = 0;
    madrona::Entity recv_free[MAX_COMM_ENTITIES_PER_NPU] {};
    int32_t recv_free_count = 0;
    madrona::Entity recv_lent[MAX_COMM_ENTITIES_PER_NPU] {};
    int32_t recv_lent_count = 0;

    inline madrona::Entity takeComm()
    {
        if (comm_free_count == 0) {
            return madrona::Entity::none();
        }
        madrona::Entity e = comm_free[--comm_free_count];
        comm_free[comm_free_count] = madrona::Entity::none();
        comm_lent[comm_lent_count++] = e;
        return e;
    }

    inline madrona::Entity takeRecv()
    {
        if (recv_free_count == 0) {
            return madrona::Entity::none();
        }
        madrona::Entity e = recv_free[--recv_free_count];
        recv_free[recv_free_count] = madrona::Entity::none();
        recv_lent[recv_lent_count++] = e;
        return e;
    }

    // Serial-context returns (inside this NPU's own process*ChakraNodes
    // instance): LIFO against the lent list, so lent stays a set.
    inline void retireComm(madrona::Entity e)
    {
        for (int32_t i = comm_lent_count - 1; i >= 0; i--) {
            if (comm_lent[i] == e) {
                comm_lent[i] = comm_lent[--comm_lent_count];
                comm_lent[comm_lent_count] = madrona::Entity::none();
                if (comm_free_count < (int32_t)MAX_COMM_ENTITIES_PER_NPU) {
                    comm_free[comm_free_count++] = e;
                }
                return;
            }
        }
    }

    inline void retireRecv(madrona::Entity e)
    {
        for (int32_t i = recv_lent_count - 1; i >= 0; i--) {
            if (recv_lent[i] == e) {
                recv_lent[i] = recv_lent[--recv_lent_count];
                recv_lent[recv_lent_count] = madrona::Entity::none();
                if (recv_free_count < (int32_t)MAX_COMM_ENTITIES_PER_NPU) {
                    recv_free[recv_free_count++] = e;
                }
                return;
            }
        }
    }
};

// Retire requests written concurrently by per-entity sys_checkFlow /
// sys_checkRecvFlow workers (atomic slot claim, like the flow inbox).
// Drained by the serial recycleCommEntities step at the frame tail.
struct NpuCommRetireReq {
    madrona::Entity entity = madrona::Entity::none();
    uint32_t is_recv = 0;
};

struct NpuCommRetireDraft {
    uint32_t count = 0;
    uint32_t overflow_count = 0;
    NpuCommRetireReq reqs[MAX_COMM_ENTITIES_PER_NPU] {};
};

// C1: liveness marker on pooled ProcessComm_E / ProcessRecvComm_E
// entities. 0 = sitting in the pool's free list (task-graph systems that
// iterate these archetypes must skip idle entities); 1 = lent out.
struct PoolSlotActive {
    int32_t active = 0;
};

// Durable FCT history for NPU-owned (dynamically injected) flows, appended
// by recordFlowCompletion() and merged into FlowCompletionBuf by the
// per-step export mirror. Unlike NpuFlowFinishedList this is never cleared
// by the system layer -- it is the network layer's own reporting channel.
struct DynamicFlowCompletionLog {
    int32_t count = 0;
    int32_t overflow_count = 0;
    FlowCompletionRecord records[MAX_DYNAMIC_FLOW_COMPLETIONS] {};
};

// Singleton snapshot of Sim scalar counts + simulationTime. Attached to
// SimDriverArch and refreshed at the end of every step by
// updateSimStatsStepSystem. mgr.cpp's GPUImpl reads it via the exported
// column pointer + cudaMemcpy so Python-visible getters work on GPU
// without needing a public Sim pointer out of MWCudaExecutor.
// 023-dtdiag: chooseDT 每步的 gap 明细, 定位 1us 钉死的赢家。
// Serial 单例里写, SimStats 镜像到 Python, digest 打印最小项。
struct DtGapLevels {
    double delayed = 0.0;
    double pending = 0.0;
    double finish = 0.0;
    double drain = 0.0;
    double backlog = 0.0;
    double pfc_pause = 0.0;
    double pfc_resume = 0.0;
    double system = 0.0;
    double chosen = 0.0;
    int32_t handoff = 0;
    int32_t drain_port = -1; // 023-dtdiag2: cachedDrainPortID 快照
    // 023-dtdiag3: 钉死端口体检 (refreshTagCounters 顺带取, serial)。
    double dport_head_bytes = 0.0;
    int32_t dport_num_chunks = 0;
    int32_t dport_dirty_alloc = 0;
    int32_t dport_dirty_clear = 0;
};
// 018-layer0-GPUdiag: per-step per-port HIGH-WATER marks for every
// FATAL-guarded fixed cap. Which FATAL fired is invisible on GPU
// (madrona::fatal varargs garble -> surfaces as cuda_exec illegal memory
// access, cf. gpu62_gdb inside fatal+944). refreshTagCounters already
// loops all ports serially each step, so max tracking is nearly free.
// Read via portMaxLevels(); compared against the MAX_* cap in digest.
struct PortMaxLevels {
    int32_t maxTags = 0;     // PortTagList.count (cap MAX_TAGS_PER_PORT=8192)
    int32_t maxOutbox = 0;   // PortOutbox.num_events (cap MAX_PORT_OUTBOX)
    int32_t maxDelayed = 0;  // PortDelayedQueue.count (cap MAX_PORT_DELAYED_EVENTS=2048)
    int32_t maxIngress = 0;  // IngressTagList.count (cap MAX_TAGS_PER_INGRESS)
    int32_t maxChunks = 0;   // PriorityBuffer.num_chunks (cap MAX_BUFFER_CHUNKS=128)
    int32_t maxWeights = 0;  // BufferChunk.num_weights (cap MAX_CHUNK_WEIGHTS)
    int32_t maxCleanup = 0;  // PortCleanup.num (cap MAX_PORT_CLEANUP)
};
struct SimStats {
    double simulationTime = 0.0;
    int32_t numFlowDefs = 0;
    int32_t numPendingFlows = 0;
    int32_t numDelayedEvents = 0;
    int32_t numActiveTags = 0;
    int32_t numSourceTags = 0;
    int32_t numFlowCompletions = 0;
    PortMaxLevels portMax = {};
    DtGapLevels dtGaps = {}; // 023-dtdiag
    // [step-trace] driver.tick at the time this mirror was written. If
    // host sees this stuck at 0 across many steps, the task graph never
    // ran scheduleStepSystem (driver.tick++) AND/OR never ran
    // updateSimStatsStepSystem. Used to distinguish "task graph silent"
    // from "HostPrint output lost".
    int32_t lastTick = 0;
};

// Flow-lifecycle singleton state. This keeps the mutable "how many flows are
// still pending / routed / completed" bookkeeping in ECS-owned storage rather
// than on the Sim world object, which makes later fan-out over FlowMeta
// entities easier to reason about.
struct FlowCounters {
    int32_t numFlowDefs = 0;
    int32_t numPendingFlows = 0;
    int32_t pendingFlowCursor = 0;
    int32_t numFlowRoutes = 0;
    int32_t numFlowCompletions = 0;
    // 021-M2: monotonic export-slot cursor for FlowCompletionBuf (static
    // flows only; dynamic flows use DynamicFlowCompletionLog). Saturates at
    // MAX_FLOW_COMPLETIONS; sim bookkeeping never gated on it (fit5 rule).
    int32_t exportCursor = 0;
};

// Step-level mutable scheduler/cache state and entity counters that were
// previously stored directly on Sim. Keeping them in an ECS singleton makes
// the remaining world-global writes explicit and prepares later fan-out of
// schedule / buffer / progress logic away from the Sim object itself.
struct SimRuntimeState {
    int32_t numDelayedEvents = 0;
    int32_t numActiveTags = 0;
    int32_t numSourceTags = 0;
    PortMaxLevels portMax = {}; // 018-layer0-GPUdiag: per-step high-waters
    DtGapLevels dtGaps = {}; // 023-dtdiag
    // Cumulative since sim start (NOT reset per frame): delayed-event drops
    // are a silent-correctness hazard (the affected flow never materializes
    // and the sim later quiesces as if finished), so the counters must
    // survive until updateSystemStatus latches SystemStatus.failed on them.
    int32_t delayedDropCount = 0;
    int32_t delayedPfcDropCount = 0;
    int32_t pfcControlEventsSeen = 0;
    int32_t progressAllExhausted = 0;
    int32_t cachedDrainPortID = -1;
    // Set when an inference-layer pipeline handoff is waiting for the next
    // frame: a KV flow completion was recorded this frame (its consumer,
    // collectInferenceKvCompletions, runs at the next frame's start) or a
    // KV-ready request was enqueued on an idle decode worker (its consumer,
    // scheduleInferenceDecode, runs at the next frame's pre-step). chooseDT
    // caps dt while this is set so the handoff frame lands right after the
    // producing frame; without the cap the consumer time quantizes to the
    // frame cadence and a distant future event (e.g. a late request
    // arrival) stretches the handoff by the whole dt. Cleared at the start
    // of every frame.
    int32_t inferenceHandoffPending = 0;
    // 025: delayed-event 活动滞后窗口 (帧数). 消费在 chooseDT 上游、
    // 重建在下游, 瞬时全空会被误判为空闲而跳到远期到达 (t2 精确值污染).
    // 帧尾 updateSystemStatus 维护: 本帧有 delayed 活动则置 8, 否则递减.
    // chooseDT 仅在窗口 >0 且 dt 为到达驱动的百毫秒级巨跳时 cap 到
    // 0.001ms (handoff 前例), 跨过瞬时窗口即释放; 单波/022 (5ms) 够不
    // 着触发条件, bit-identical.
    int32_t delayedActiveWindow = 0;
    double cachedNextDelayedGap = 0.0;
    double cachedNextBacklogGap = 0.0;
    double cachedNextPfcPauseGap = 0.0;
    double cachedNextPfcResumeGap = 0.0;
    double cachedNextDrainTime = 0.0;
    double cachedNextFinishTime = 0.0;
    double nextDT = 0.0;
};

enum class StepPhaseID : uint32_t {
    Schedule = 0,
    Deliver,
    Ingress,
    Alloc,
    PfcEmit,
    ClearDT,
    BufferProgress,
    NumPhases,
};

struct StepPhaseTimes {
    int32_t step = 0;
    double totalWallTimeS = 0.0;
    double phaseWallTimeS[(uint32_t)StepPhaseID::NumPhases] {};
    double _stepStartWallTimeS = 0.0;
    double _lastBoundaryWallTimeS = 0.0;
};

// Singleton mirror of Sim::flowCompletions[].record for GPU export. Same
// fixed capacity as MAX_FLOW_COMPLETIONS; entries past numFlowCompletions
// are zero-initialised and should not be read.
struct FlowCompletionBuf {
    FlowCompletionRecord records[MAX_FLOW_COMPLETIONS] {};
};

struct SimDriverArch : public madrona::Archetype<
    SimDriver
> {};

// Event structs (moved from sim.hpp so PortOutbox can embed them in the
// Port archetype). Kept POD so the Port component array stays fixed-size
// and GPU-friendly.
struct FlowArrivalEv {
    int32_t port_id = -1;
    FlowId flow_id = -1;
    Bytes size = 0.0;
    Bw in_bw = 0.0;
    int32_t is_source = 0;
    int32_t priority = 0;
};

struct BwUpdateEv {
    int32_t port_id = -1;
    FlowId flow_id = -1;
    Bw in_bw = 0.0;
};

struct PfcControlEv {
    int32_t target_port_id = -1;
    int32_t source_port_id = -1;
    int32_t priority = 0;
    int32_t paused = 0;
};

struct DelayedEvent {
    enum class Type : int32_t {
        Arrival,
        BwUpdate,
        PfcControl,
    } type = Type::Arrival;

    Time t = 0.0;
    FlowArrivalEv arrival {};
    BwUpdateEv bwupd {};
    PfcControlEv pfcctrl {};
};

// --- A7 (NPU flow schedule) draft pattern ----------------------------------
// Parallel per-NPU draft for Sim::scheduleNpuFlows' split: each NPU walks
// its own NpuFlowActiveList, does every flow-local check/write (pending/
// active flags, route materialization), and records a push ticket (a copy
// of the prepared DelayedEvent) for flows with prepared != 0. The serial
// commit inside flushScheduleStepSystem replays tickets in npu_id
// ascending order, re-evaluating the global numDelayedEvents capacity gate
// per ticket exactly where the legacy loop evaluated it, so the global
// delayed-event registration order (F9 invariant) is preserved byte for
// byte. numFlowRoutes increments are deferred to the commit via the
// route_new flag so even the trace-log-visible counter timeline matches
// the legacy loop.
struct NpuScheduleTicket {
    DelayedEvent event {};
    int32_t route_new = 0;
};

struct NpuFlowScheduleDraft {
    uint32_t count = 0;
    uint32_t overflow_count = 0;
    NpuScheduleTicket tickets[MAX_FLOWS_PER_NPU] {};
};

// --- 013-bug修复 (M1) create-draft pattern ----------------------------------
// Parallel per-NPU draft: createFlowsFromNpuRequestsOneNpu materializes the
// FlowDef/Route/Runtime/Schedule components on its OWN pool entities (safe)
// but must NOT touch the shared flowMetaLookupIds/EntityLookup hash table:
// parallel inserts from concurrent NPU instances can land on overlapping
// linear-probe paths (lost insert) and the ids/entity double-store tears.
// Each instance therefore records (flow_id, entity) tickets here, in the
// inbox-sorted order it already processes; the serial commitNpuFlowSchedule
// companion (commitNpuFlowCreate, replayed in npu_id ascending order right
// after the create fan-out) inserts them single-threaded. Ticket count per
// step per NPU <= inbox.count <= MAX_FLOWS_PER_NPU, so overflow is
// structurally unreachable; it still fails loudly per the A1/A7 design.
struct NpuFlowCreateTicket {
    FlowId flow_id = -1;
    madrona::Entity entity = madrona::Entity::none();
};

struct NpuFlowCreateDraft {
    uint32_t count = 0;
    uint32_t overflow_count = 0;
    NpuFlowCreateTicket tickets[MAX_FLOWS_PER_NPU] {};
};

enum class NodeType : int32_t {
    Host,
    Switch,
};

struct DirtyPort {
    int32_t isDirty = 0;
};

struct PortState {
    int32_t port_id = -1;
    NodeId node_id = -1;
    int32_t port_idx = -1;
    Bw port_bw = 0.0;
    int32_t connected = 0;
    int32_t next_port_id = -1;
};

struct FlowTagState {
    int32_t port_id = -1;
    FlowId flow_id = -1;
    int32_t priority = 0;
    Bw in_bw = 0.0;
    Bw out_bw = 0.0;
    Bw prev_out_bw = 0.0;
    Bytes backlog = 0.0;
    Time last_backlog_time = 0.0;
    Bytes remaining = 0.0;
    Time last_remaining_time = 0.0;
    int32_t is_source = 0;
    int32_t downstream_created = 0;
    int32_t next_port_id = -1;
    int32_t ingress_port_id = -1;
    // Phase D: cached owning port entity so destroyTag can remove this tag
    // from PortTagList without rescanning portEntities[]. Written by
    // createTagOnPort, cleared on tag destruction.
    madrona::Entity port_entity = madrona::Entity::none();
};

// Per-tag scratch used by the parallel progress scan. The owning Port worker
// marks finished source tags here and the singleton cleanup pass later
// destroys only the tags queued by each port.
struct FlowTagProgress {
    int32_t pending_source_destroy = 0;
};

struct FlowWeight {
    FlowId flow_id = -1;
    double weight = 0.0;
};

struct BufferChunk {
    Bytes chunk_bytes = 0.0;
    int32_t num_weights = 0;
    FlowWeight weights[MAX_CHUNK_WEIGHTS] {};
};

struct PriorityBuffer {
    Bytes buf_cnt = 0.0;
    int32_t num_chunks = 0;
    int32_t head = 0;
    int32_t tail = 0;
    BufferChunk buf_chunks[MAX_BUFFER_CHUNKS] {};
    double net_buffer_rate = 0.0;
};

struct PortBuffer {
    PriorityBuffer prior_bufs[PFC_MAX_PRIORITY] {};
    Time last_update_time = 0.0;
};

struct PortPfcConfig {
    double xoff[PFC_MAX_PRIORITY] {};
    double xon[PFC_MAX_PRIORITY] {};
    int32_t pfc_enabled = 0;
};

struct PortPfcState {
    int32_t paused[PFC_MAX_PRIORITY] {};
    int32_t pause_active[PFC_MAX_PRIORITY] {};
    // Consecutive frames this port has been pause_active while outside the
    // dirty-derived check-target set (starvation watchdog, sim_systems_pfc).
    int32_t stalled_frames = 0;
    // [002-pfc-instrumentation] Per-pause-episode frame accounting, reset
    // whenever the port is not paused and printed by printPfcEpisodeLog at
    // the resume detect. Answers "was the paused port actually being
    // re-checked (dirty mark) or did it wait on the watchdog".
    int32_t ep_frames_paused = 0;
    int32_t ep_frames_marked_dirty = 0;
    int32_t ep_frames_marked_watchdog = 0;
    int32_t ep_frames_unmarked = 0;
    // Next sim-time (ms) at which to emit a [PFCSAMP] serving-state sample
    // while this port stays paused; 0 = sample on the first qualifying frame.
    double ep_next_sample = 0.0;
    int32_t paused_upstream_count[PFC_MAX_PRIORITY] {};
    int32_t paused_upstreams[PFC_MAX_PRIORITY][MAX_PAUSED_UPSTREAMS] {};
    int32_t pfc_cnt[PFC_MAX_PRIORITY] {};
    double last_fill_rate[PFC_MAX_PRIORITY] {};
    int32_t pfc_check_target = 0;
    // Phase C: deferred PFC timer mutations. The per-Port pfcDetectOnePort
    // sets want_* flags on its own PortPfcState; the flushPortPfcTimers
    // singleton applies clear-then-set to Sim::pfc{Pause,Resume}Timers in
    // port_id ascending order so the global timer arrays stay race-free
    // on GPU and deterministic across backends.
    int32_t want_clear_pause = 0;
    int32_t want_clear_resume = 0;
    int32_t want_set_pause = 0;
    int32_t want_set_resume = 0;
    double set_pause_t = 0.0;
    double set_resume_t = 0.0;
};

// --- Per-port scratch / hint state introduced in phase B. ---
// These remain POD so they can live either on the Port archetype or in
// Sim-owned port_id indexed arrays. Cross-port reductions, deferred entity
// destruction, and deferred log emission consume these fields in a
// deterministic singleton pass driven by SimDriverArch.

struct PortCachedHints {
    int32_t has_drain_hint = 0;
    int32_t has_finish_hint = 0;
    int32_t has_active_finish = 0;
    double drain_hint_t = 0.0;
    double finish_hint_t = 0.0;
    double active_finish_t = 0.0;
};

struct PortDrainHint {
    int32_t want_clear = 0;
    int32_t want_set = 0;
    double set_t = 0.0;
};

struct PortTimers {
    Time backlog_drain = 0.0;
    Time pfc_pause = 0.0;
    Time pfc_resume = 0.0;
};

struct PortCleanup {
    int32_t num = 0;
    madrona::Entity tags[MAX_PORT_CLEANUP] {};
    int32_t propagate[MAX_PORT_CLEANUP] {};
};

struct PortFinishedSourceList {
    int32_t num = 0;
    madrona::Entity tags[MAX_PORT_CLEANUP] {};
};

struct PortTraceLast {
    // alloc summary (populated only when this port had at least one live tag).
    int32_t has_alloc_trace = 0;
    double alloc_port_bw = 0.0;
    int32_t alloc_num_tags = 0;
    int32_t alloc_num_live = 0;
    double alloc_sum_in = 0.0;
    double alloc_sum_out = 0.0;
    int32_t alloc_is_dest_only = 0;
    int32_t alloc_has_buffer = 0;
    // buffer summary.
    int32_t buffer_processed = 0;
    int32_t buffer_destroy_count = 0;
    double buffer_total_buf_cnt = 0.0;
    // Phase C: per-Port emit (downstream) summary.
    int32_t emit_is_dirty = 0;
    int32_t emit_arrival_count = 0;
    int32_t emit_bwupdate_count = 0;
    // Phase C: per-Port pfcDetect summary. emitted_pfc counts events pushed
    // into PortOutbox; checked counts ports that actually ran the detect
    // body (i.e. participated in the emit_pfc summary "checked_port_count").
    int32_t pfc_detect_checked = 0;
    int32_t pfc_detect_emitted = 0;
    int32_t pfc_detect_dropped = 0;
    int32_t deliver_arrival_count = 0;
    int32_t deliver_bwupdate_count = 0;
    int32_t deliver_pfc_count = 0;
    int32_t deliver_pfc_dropped = 0;
    int32_t progress_source_scan_count = 0;
    int32_t progress_finished_source_count = 0;
    int32_t progress_emitted_cleanup_count = 0;
    int32_t progress_buffered_dirty_marked = 0;
    // Snapshot of DirtyPort captured at alloc time (before emit/clear run);
    // consumed by the cachedNextDrainTime reduction singleton. Populated by
    // allocOnePort once phase B.2 lands.
    int32_t was_dirty_at_alloc = 0;
    // Snapshot of DirtyPort captured by clearDirtyOnePort (after emit).
    // advanceOnePortBuffer consumes this directly on the owning Port entity
    // instead of folding it back into a Sim-global dirty-port list.
    int32_t was_dirty_at_clear = 0;
    // Phase B.1 per-port runtime-scan scratch, reduced by snapshotDirtyPorts.
    int32_t clear_has_delayed_gap = 0;
    double clear_delayed_gap = 0.0;
    int32_t clear_has_backlog_gap = 0;
    double clear_backlog_gap = 0.0;
    int32_t clear_has_pfc_pause_gap = 0;
    double clear_pfc_pause_gap = 0.0;
    int32_t clear_has_pfc_resume_gap = 0;
    double clear_pfc_resume_gap = 0.0;
    // Phase E: ingress_chain per-Port summary counters. The per-Port
    // arrival/bwUpdate/pfcPropagate workers increment these; the
    // logIngressChainSystem singleton emits the single summary line
    // aggregated across all ports (same format as the legacy
    // printSystemArrivalSummary / printSystemBwUpdateSummary /
    // printSystemPfcSummary). The per-event detail log lines (create /
    // update / destroy / buffered_zero / forward / complete) are still
    // emitted from within the per-Port worker; scope "ingress_chain" is
    // canonical-sorted, so per-port interleaving is tolerated.
    int32_t arrival_created = 0;
    int32_t arrival_updated = 0;
    int32_t arrival_skipped = 0;
    int32_t bwupd_created = 0;
    int32_t bwupd_updated = 0;
    int32_t bwupd_buffered_zero = 0;
    int32_t bwupd_destroyed = 0;
    int32_t bwupd_forwarded = 0;
    int32_t bwupd_completed = 0;
    int32_t bwupd_skipped = 0;
    int32_t pfc_applied = 0;
    int32_t pfc_skipped = 0;
};

struct PortOutbox {
    int32_t num_events = 0;
    DelayedEvent events[MAX_PORT_OUTBOX] {};
};

struct PortTagLookupEntry {
    FlowId flow_id = -1;
    madrona::Entity entity = madrona::Entity::none();
};

struct PortTagLookup {
    PortTagLookupEntry entries[MAX_PORT_TAG_LOOKUP] {};
};

struct PortDelayedQueue {
    int32_t head = 0;
    int32_t count = 0;
    DelayedEvent events[MAX_PORT_DELAYED_EVENTS] {};
};

struct PortTagList {
    int32_t count = 0;
    madrona::Entity tags[MAX_TAGS_PER_PORT] {};
};

// Source-only mirror for the same port-owned tag set. progressFinishedSources
// uses this to skip the per-tag is_source filter on the hot path. If a port
// ever exceeds the fixed local capacity, overflow latches and the caller
// falls back to PortTagList to preserve correctness.
struct PortSourceTagList {
    int32_t count = 0;
    int32_t overflow = 0;
    madrona::Entity tags[MAX_TAGS_PER_PORT] {};
};

// Authoritative per-ingress-port tag list used by ingress-scoped systems.
// This is kept on the Port entity so ingress-side work can stay entity-local
// instead of bouncing through a shared global mirror.
struct IngressTagList {
    int32_t count = 0;
    int32_t overflow = 0;
    madrona::Entity tags[MAX_TAGS_PER_INGRESS] {};
};

struct PortInbox {
    int32_t num_arrival = 0;
    FlowArrivalEv arrivals[MAX_PORT_INBOX_ARRIVAL] {};
    int32_t num_bwupd = 0;
    BwUpdateEv bwupds[MAX_PORT_INBOX_BWUPD] {};
    int32_t num_pfc = 0;
    PfcControlEv pfcs[MAX_PORT_INBOX_PFC] {};
};

// Phase E: deferred tag-create requests. Per-Port arrival/bwUpdate
// workers push local create requests here; a later per-Port materialize
// pass consumes them via createTagOnPort without directly touching any
// other port's state.
struct PortCreateReq {
    int32_t from_arrival = 0;
    FlowId flow_id = -1;
    Bw in_bw = 0.0;
    Bytes size = 0.0;
    int32_t is_source = 0;
    int32_t priority = 0;
    int32_t log_enabled = 0;
    const char *log_label = nullptr;
};

struct PortCreateList {
    int32_t num = 0;
    PortCreateReq reqs[MAX_PORT_CREATE] {};
};

// Per-port free list of preallocated FlowTag entities. This removes
// runtime ctx.makeEntity/destroyEntity traffic from the hot path and is the
// prerequisite for later making tag creation/destruction fully per-port.
struct PortTagPool {
    int32_t free_count = 0;
    int32_t allocated_count = 0;
    madrona::Entity free_tags[MAX_TAGS_PER_PORT] {};
};

// Cross-port ingress-list links deferred out of createTagOnPort. The
// per-Port tag-create materialize worker records "append this new tag to
// ingress port X", and a tiny singleton later replays those appends in
// port_id order so IngressTagList remains deterministic and race-free.
struct PortIngressLinkReq {
    int32_t ingress_port_id = -1;
    madrona::Entity tag_entity = madrona::Entity::none();
};

struct PortIngressLinkList {
    int32_t num = 0;
    PortIngressLinkReq reqs[MAX_PORT_INGRESS_LINKS] {};
};

// Deferred dirty-port fanout. Per-port workers that discover "these
// owning ports must become dirty" append target port ids here; a tiny
// singleton later replays the writes to DirtyPort in port_id order.
struct PortDirtyMarkList {
    int32_t num = 0;
    int32_t port_ids[MAX_PORT_DIRTY_MARKS] {};
};

// Phase E: deferred recordFlowCompletion requests. bwUpdate on a
// terminal port must not mutate Sim::flowCompletions / flowDefs /
// flowRoutes directly; it pushes the flow_id here and
// flushFlowCompletionSystem applies them in port_id ascending order.
struct PortCompletionReq {
    FlowId flow_id = -1;
    Time end_time = 0.0;
};

struct PortCompletionList {
    int32_t num = 0;
    PortCompletionReq reqs[MAX_PORT_COMPLETE] {};
};

struct PortIngressUnlinkReq {
    int32_t ingress_port_id = -1;
    madrona::Entity tag_entity = madrona::Entity::none();
};

struct PortIngressUnlinkList {
    int32_t num = 0;
    PortIngressUnlinkReq reqs[MAX_PORT_INGRESS_UNLINKS] {};
};

// GPU stack relief (019/018-layer0, 2026-10-23): per-port scratch moved
// off the per-thread stack into a per-port component. Background: capacity
// bumps (MAX_TAGS_PER_PORT 1024->4096, MAX_CHUNK_WEIGHTS 512->4096)
// enlarged thread-local scratch arrays in the alloc/buffer path; the NVRTC
// megakernel per-thread local grew 164824B (003) -> 393768B and
// cuGraphInstantiate's stack reservation (local x 129024 threads = 50.8GB)
// exceeded the 46GB card even for 2-rank micro. Component storage costs
// ~310KB/port of device-global (not per-thread) memory and leaves the CPU
// logic statement-for-statement identical. All array reads are either
// bounded by counts that are reset at each phase entry (replacing the old
// fresh-local zero-init semantics -- see the resets in allocOnePort /
// allocOnePortQoS* / advanceOnePortBuffer) or write-before-read scratch
// (build_chunk/gc_chunk/head_ids, zeroed at each build site like before).
struct PortAllocScratch {
    madrona::Entity tags[MAX_TAGS_PER_PORT] {};
    madrona::Entity live_tags[MAX_TAGS_PER_PORT] {};
    int32_t num_tags = 0;
    int32_t num_live = 0;
    double live_sum_in = 0.0;
};

struct PortQosBuckets {
    madrona::Entity tags[PFC_MAX_PRIORITY][MAX_TAGS_PER_PORT] {};
    int32_t counts[PFC_MAX_PRIORITY] {};
    double in_sum[PFC_MAX_PRIORITY] {};
};

struct PortFrameScratch {
    PortAllocScratch alloc {};
    PortQosBuckets qos {};
    madrona::Entity adv_tags[MAX_TAGS_PER_PORT] {};
    madrona::Entity adv_pri_tags[PFC_MAX_PRIORITY][MAX_TAGS_PER_PORT] {};
    int32_t adv_pri_counts[PFC_MAX_PRIORITY] {};
    double adv_pri_in[PFC_MAX_PRIORITY] {};
    double adv_pri_out[PFC_MAX_PRIORITY] {};
    BufferChunk build_chunk {};
    BufferChunk gc_chunk {};
    int32_t head_ids[MAX_CHUNK_WEIGHTS] {};
};

struct Port : public madrona::Archetype<
    PortState,
    PortBuffer,
    DirtyPort,
    PortCleanup,
    PortFinishedSourceList,
    PortOutbox,
    PortTagLookup,
    PortDelayedQueue,
    PortTagList,
    PortSourceTagList,
    PortInbox,
    PortCreateList,
    PortTagPool,
    PortIngressLinkList,
    PortIngressUnlinkList,
    PortDirtyMarkList,
    PortCompletionList,
    PortPfcConfig,
    PortPfcState,
    PortCachedHints,
    PortDrainHint,
    PortTimers,
    PortTraceLast,
    IngressTagList,
    PortFrameScratch
> {};

struct FlowTag : public madrona::Archetype<
    FlowTagState,
    FlowTagProgress
> {};

}
