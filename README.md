# Madrona 流级 LLM 训练/推理仿真器

> 基于 [Madrona ECS 引擎](https://madrona-engine.github.io) 的 LLM 集群数据中心网络仿真器。
> 底层是一个**流级 (flow-level) 网络仿真器**，上层融合了 **Chakra 系统层**（任务依赖 + 集合通信调度）。
> 目前完整支持 **训练负载仿真**；**推理仿真（PD 分离 / KV Transfer）** 处于方案规划阶段，复用同一执行引擎。

---

## 1. 一句话介绍

这个仓库把"网络仿真"和"系统仿真"焊在了一起：

- **网络层**：把一个通信流沿路径拆成端口上的 `FlowTag`，事件驱动 + 变步长 (`chooseDT`) 推进，支持 PFC 流控、带宽分配、backlog 排队——精确算每个流的完成时间 (FCT)。
- **系统层**：解析每个 NPU 的 **Chakra 任务 DAG**（COMP / COMM_SEND / COMM_RECV / COMM_COLL），按依赖关系调度算子，把通信任务通过 `net_sys_interface` 的 `setFlow()` 下发给网络层，再用 `checkFlowFinish()` 回收完成事件。

两层共享同一个仿真时钟 `Sim::now`，最终输出**所有 NPU 都完成的系统级结束状态**和**每条流的 FCT 分布**。

---

## 2. 业务背景与当前能力

| 场景 | 状态 | 说明 |
| --- | --- | --- |
| **训练仿真** | ✅ 已支持 | 输入 Chakra 训练 workload（如 deepseek-671B 2TP/4EP/2PP 训练） + 拓扑文件，仿真到全部 NPU 完成 |
| **推理仿真** | 🔜 规划中 | 新增 inference runtime 层（Prefill/Decode 分离、PD Router、KV Transfer），DAG 遍历与网络 DES 完全复用 |

### 2.1 训练仿真是怎么跑的

1. Python 读取 **Chakra workload**（每个 NPU 一个 JSON 任务 DAG）和 **拓扑文件**（NODE/LINK/FLOW）。
2. 系统层把 Chakra 节点解析进每 NPU 的 `ChakraNodes`，每一帧找出**依赖已解除的 ready 节点**执行：
   - COMP 节点：占用硬件资源一段时间（`durationMicros`）；
   - COMM 节点：按 Ring 集合通信语义把通信任务拆成收发流，调用 `setFlow()` 下发给网络层；
   - 节点完成后解除下游依赖，直到每 NPU 的 DAG 全部跑完。
3. 网络层负责把这些流真实地跑过交换机拓扑，算出完成时间。
4. 仿真结束条件：**每个 NPU 的 DAG 都执行完毕**（`SystemStatus.finished == 1`）。

### 2.2 推理仿真要做什么（规划中）

推理与训练的执行结构相同（都是 forward 算子依赖图），只是算子的 `duration` / `comm_size` 从**静态 trace** 变为**由 inference batch 动态解析**。规划方案是：

```
原来训练:  ready node → 读 trace 里的 duration/comm_size → 创建 task
推理:      ready node → 查 inference batch context → 动态算 duration/comm_size → 创建 task
                          ↑
                  新增 inference runtime 层（PD Router / Continuous Batching / KV Transfer Planner）
```

KV Transfer（P 池 prefill 完成后把 KV cache 发给 D 池）就是一次普通的 `setFlow()` 调用，直接复用网络层。

详见 `report/3-推理仿真/0.推理背景与整体逻辑.md` 与 `report/3-推理仿真/# Multiverse LLM Serving Simulation Design_v2.md`。

---

## 3. 架构总览

```
                        ┌──────────────────────────────────────────────┐
  Python 输入             │                 Madrona ECS (CPU/GPU)          │
 ┌──────────────┐        │  ┌────────────────────────────────────────┐  │
 │ topo 文件     │──┐     │  │        系统层 (Chakra 调度)             │  │
 │ chakra workload│─┤     │  │  NpuNode / ChakraNodes / HardwareResource│ │
 │ 配置参数       │  │     │  │  sys_processChakraNodes → ready 节点     │  │
 └──────────────┘  │     │  └───────────────┬────────────────────────┘  │
                   │     │        setFlow() │ checkFlowFinish()          │
   bindings.cpp    │     │   ┌──────────────▼────────────────────────┐  │
   mgr.cpp         │     │   │    网络层 (流级网络仿真)               │  │
   (GridState/     │──┼──▶│   │  Port / FlowTag / FlowMeta            │  │
    NetworkInit/   │     │   │  事件驱动 + chooseDT 变步长 + PFC       │  │
    Chakra tensor) │     │   └────────────────────────────────────────┘  │
 └──────────────┘        │               │                              │
                        └───────────────┼──────────────────────────────┘
                                        ▼
                        SystemStatus (finished) + FlowCompletion (FCT) CSV
```

- **耦合点只有一个**：`src/sys/net_sys_interface.hpp` 里的接口（`getCurrentTime` / `isExistedFlow` / `addSimtime` / `setFlow` / `checkFlowFinish`）。
- **时间单位**：网络层内部 `double ms`，接口对外 `uint64_t ns`。
- **数据接入两条路**（详见 `PYTHON_TO_GPU_DATAFLOW.md`）：
  - 大块共享只读数据（topo / flow）：`bindings → Manager → cudaMemcpy 共享块 → WorldInit → Sim`；
  - 每步动态字段（Chakra 数据 / 参数）：`exportColumn + to_torch()` tensor。

---

## 4. 网络层业务逻辑（流级仿真器）

### 4.1 核心建模

- **一个流不是整条算**，而是拆成沿路径每个端口上的 `FlowTag`。
- 每个 `FlowTag` 记录：输入带宽 `in_bw`、输出带宽 `out_bw`、积压 `backlog`、源端剩余量 `remaining`。
- 端口之间靠三类**延迟事件**传播：`Arrival`（新流到达）、`BwUpdate`（带宽变化）、`PfcControl`（流控）。
- **变步长**：每帧 `chooseDT()` 取"下一最近事件时间"作为 `dt`，而非固定步长。

### 4.2 每帧 12 阶段流水线（任务图见 `src/sim_tasks.cpp`）

1. **Schedule**：`tick++`；`schedulePendingFlows()` 把到启动时间的流注入为首个 Arrival；NPU 动态请求实体化（`createFlowsFromNpuRequests` / `scheduleNpuFlows`）。
2. **Deliver**：`deliverEvents()` 把到时事件分发进各端口的 `PortInbox`。
3. **Ingress 链**（按端口并行）：`pfcPropagate` → `flowArrival`（登记新 tag）→ `bwUpdate`（更新/清理/转发/完成登记）→ 统一 `flushTagCreate / flushTagCleanup / flushFlowCompletion / flushPortOutbox`。
4. **Alloc**（带宽分配）：按 QoS（无/SP/WRR）、PFC 状态、backlog 给每个 tag 算 `out_bw`，产出 drain/finish 预测 hint。
5. **PFC Detect / Emit**：阈值检测决定是否给上游发 pause/resume；`emitOnePort` 把带宽变化传播给下游（首跳发 Arrival，之后仅 `out_bw` 变化时发 BwUpdate）。
6. **Clear / ChooseDT**：清 dirty 快照，计算下个 `dt`。
7. **Buffer 推进**：`advanceOnePortBuffer` 推进 backlog/chunk；`flowProgressAndCleanup` 处理源流完成、计时器到期、`now += dt`。
8. **系统层尾节点**：`sys_checkNpuFinish` / `updateSystemStatus` 判定整体完成。

### 4.3 一个流的完整旅程

```
start_time 到 → schedule 注入首个 Arrival → deliver 投给首跳端口
→ flowArrival + createTag 生成源 tag → alloc 算 out_bw → emit 沿路由发向下游
→ 每跳重复"到达→分配→发射"，拥塞时 backlog 增大、PFC 触发 pause 缓解后 resume
→ 源端 remaining 清零 → 沿路径清掉下游残留 tag → recordFlowCompletion 写 FCT
```

---

## 5. 系统层业务逻辑（Chakra 调度）

系统层代码在 `src/sys/`（从 `multiverse-dev` 移植，命名空间 `madsimple::llm_system`）：

| 文件 | 职责 |
| --- | --- |
| `sys_types.hpp` | 系统层组件：`NpuNode` / `ChakraNodes` / `HardwareResource` / `ProcessingCompTask` / `ProcessingCommTasks` / `NpuFlowInbox` 等 |
| `chakra_node_processing.cpp` | 把 `ChakraNodesData` tensor 解析为每 NPU 的 `ChakraNode[]`，找无依赖节点 |
| `llm_system.cpp` | 核心 `sys_processChakraNodes()`：每帧推进每个 NPU 的 DAG，执行 COMP/COMM 节点、解除依赖（已被 `MADRONA_NO_INLINE` + 拆分控制编译规模） |
| `communication_system.cpp` | Ring 集合通信：`sys_checkFlow`（查发送完成）、`sys_checkRecvFlow`（配接收完成） |
| `time_management.cpp` | SkipTime 时间管理、节点移除、`sys_checkNpuFinish` |
| `net_sys_interface.cpp` | 系统层 ↔ 网络层 5 个解耦接口的实现 |

### 关键机制：per-NPU 无锁邮箱

每个 NPU 实体持有自己的 `NpuFlowInbox / NpuFlowPool / NpuFlowActiveList / NpuFlowFinishedList / NpuFlowPairState`：

- 通信节点调用 `setFlow()` 时，只写**自己 NPU 的 inbox**（同一 worker 内，无跨 block 竞争，无需锁）；
- `createFlowsFromNpuRequests` 串行单例每帧从各 NPU inbox 弹出请求，从该 NPU 自己的实体池取出 `FlowMeta` 填充；
- SEND 完成写入目标 NPU 的 `NpuFlowPairState` 邮箱，RECV 系统原子消费；
- 这样任何一个 NPU 的"未 ready 流"都不会阻塞其他 NPU 的 ready 流（消除 head-of-line blocking）。

---

## 6. 业务代码在哪里（代码地图）

### 6.1 C++ 网络层（`src/`）

| 文件 | 职责 |
| --- | --- |
| `types.hpp` | 组件 / archetype / 容量常量：`Port` / `FlowTag` / `FlowMeta` / `SimDriverArch` / 各 inbox/邮箱 |
| `sim.hpp` | `Sim` world 类型：拓扑数组、路由表、`Sim::*` 方法声明、`Sim::Config` 配置 |
| `sim_tasks.cpp` | **主循环入口**：`setupTasks()` 组装 12 阶段任务图 + 系统层节点，最好从这里看起 |
| `sim_init*.cpp` | 初始化：`resetNetworkState` / `loadTopo` / `loadFlow` / `createPort` |
| `sim_systems_*.cpp` | 网络层各阶段系统：schedule / dispatch / ingress / bandwidth / buffer / pfc / progress / lifecycle |
| `sim.cpp` | `registerTypes` + `Sim` 构造函数 |
| `bindings.cpp` / `mgr.cpp` | 宿主接口：Python 参数 → `Manager` → CPU/GPU 执行器、导出 tensor |

### 6.2 C++ 系统层（`src/sys/`）

见第 5 节表格。

### 6.3 Python 层（`src/madrona_simple_example/`）

| 文件 | 职责 |
| --- | --- |
| `gridworld.py` | `GridWorld` 包装类：解析 topo/flow 文件、上传 Chakra 数据、step 驱动、读状态 |
| `chakra/` | Chakra JSON 解析与序列化（`parser.py` / `conversion.py` / `config.py`） |

### 6.4 运行脚本（`scripts/`）

| 脚本 | 用途 |
| --- | --- |
| `run_flow_simulator.py` | 端到端跑一个 topo + workload，可输出 FCT CSV |
| `run_parity.py` | CPU/GPU 一致性对比 |
| `compute_chakra_capacity.py` | 按 workload 自动算 `CHAKRA_NODES_DATA_LENGTH` 等编译期容量 |
| `run.py` | 原 GridWorld 最小示例（跑 Agent/网格） |

根目录 `run_flow_simulator.sh` / `run_flow_simulator_gpu.sh` / `run_flow_simulator_cpu_log.sh` 是实际的启动入口（会先自动改写 `sys_types.hpp` / `sys_config.hpp` / `chakra/config.py` 里的 NPU 数、容量和日志开关，再构建运行）。

---

## 7. 构建与运行

依赖：CMake ≥ 3.18、Python、CUDA（GPU 路径固定用 **CUDA 12.8**），`external/madrona` 是子模块。

```bash
# 拉取（含子模块）
git clone --recursive <repo> && cd madrona_simple_example

# 构建（CPU 后端 + Python 绑定）
mkdir -p build && cmake -S . -B build && cmake --build build -j$(nproc)
pip install -e .

# 最小示例（原 GridWorld）
python scripts/run.py 32            # CPU
python scripts/run.py 32 --gpu      # GPU

# 端到端流仿真（CPU）
python scripts/run_flow_simulator.py \
    --topo /app/jiuding_dodsim/examples/leafspine128/leafspine_h128_topo.txt \
    --workload <chakra_workload_dir> \
    --npu-count 16 --ring-dims 2,2,4 \
    --max-steps 200000 --print-every 10 \
    [--gpu] [--out-csv out.csv]

# GPU 推荐用包装脚本（自动配置容量 / CUDA 环境 / 内核缓存 / device heap）
bash run_flow_simulator_gpu.sh
```

> 注意：`run_flow_simulator_gpu.sh` 会把 `sys_types.hpp` 等源码文件里的编译期常量用 sed 改写，并在 `.cache/` 生成带指纹的 megakernel 缓存。GPU 首次冷编译很慢（巨型系统函数），后续命中缓存。CUDA 版本 / 源码 / 容量任何变化都会使缓存失效。

### 关键运行参数（`Sim::Config` / 脚本变量）

- 网络：`enable_pfc` / `pfc_egress` / `pfc_xoff_threshold` / `pfc_xon_threshold` / `qos_mode`（0/SP/WRR）/ `prior_weights` / `dt_min` / `propagation_interval`。
- 系统层：`npu_count` / `ring_dims`（如 2,2,4）/ `chunks_num` / `max_steps` / `print_every`。
- GPU 调试：`GPU_COMPILE_BOOTSTRAP`（快速编译）、`GPU_TRACE_INIT`、`GPU_DEVICE_HEAP_SIZE`（默认 64MB，见第 8 节坑位）。

---

## 8. 已知工程坑位（踩坑记录）

详细过程见 `report/优化gpu编译速度/` 系列文档：

1. **巨型函数拖垮 GPU 冷编译**：`sys_processChakraNodes()` 曾经 831 行且被 LTO 内联进 megakernel，NVRTC/nvJitLink 优化数小时不结束。已拆分为按节点类型的 helper + `MADRONA_NO_INLINE`。
2. **CUDA 12.8 的 LLVM 元数据错误**：`cuda::atomic` + 设备 LTO + `-lineinfo` 组合触发 `Broken module`。已在 `external/madrona/src/mw/cuda_exec.cpp` 关掉 Optimize 路径的 `-dlto` / `-lineinfo`。
3. **Device heap 4GB → 64MB**：默认 4GB 设备 heap 在"第一个使用设备 malloc 的内核"（`constructGraphs`）launch 时一次性预留 4GB 连续地址空间失败 → `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`。调小后即可运行（详见 `report/优化gpu编译速度/6.device_heap与启动失败时间线解析.md`）。
4. **GPU 初始化时序**：`initECS / initWorlds / initTasks` 必须始终同步，否则初始化不稳定。
5. **CPU/GPU 结果分叉**：同 NPU 多个通信节点并发写 inbox 时普通 `count++` 会丢流，已改为原子领取槽位 + 串行排序消费。

---

## 9. 文档索引

| 文档 | 内容 |
| --- | --- |
| `MADRONA_BUSINESS_GUIDE.md` | 业务开发方法：组件 / 实体 / 系统 / 任务图四件套怎么写（Madrona 业务编码规范） |
| `PYTHON_TO_GPU_DATAFLOW.md` | Python → GPU 两条数据通路（exportColumn vs 共享 GPU 结构体） |
| `TOPO_FLOW_INPUT_PLAN.md` | 拓扑/流量外部输入改造方案 |
| `sim拆分方案.md` | `sim.cpp` 按业务阶段拆分方案（当前文件布局的来源） |
| `learn/` | 流仿真器业务讲解、系统流转示意、实体-系统矩阵 |
| `report/2-merge-plan/` | 系统层 + 流仿真器合并方案（4 阶段 A/B/C/D） |
| `report/3-推理仿真/` | 推理仿真背景、PD 分离、设计文档 |
| `report/优化gpu编译速度/` | GPU 编译与运行故障完整复盘（含 device heap 解析） |
| `report/4-input-plan/` | 华为负载输入解析、flow_id 对比等 |

---

## 10. 快速上手建议

1. 先读 `learn/业务讲解文档.md`（网络层）和 `report/2-merge-plan/2.方案.md`（两层融合）。
2. 再看 `src/sim_tasks.cpp` 的 `setupTasks()`，理解一帧的完整顺序。
3. 用 `scripts/run_flow_simulator.py` 跑一个小 workload，对比 CPU / GPU 输出。
4. 需要改业务时按 `MADRONA_BUSINESS_GUIDE.md` 的规范，在 `types.hpp` / `sim_systems_*.cpp` / `sys/` 三个区域动手。