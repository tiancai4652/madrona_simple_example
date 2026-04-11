# Madrona 拓扑 / 流量外部输入改造方案

## 1. 背景与目标

当前 `madrona_simple_example` 的拓扑与流量数据仍然是硬编码在 C++ 里的，主要位于：

- `madrona_simple_example/src/sim_init.cpp`
  - `buildHardcodedTopo()`
  - `buildHardcodedFlows()`

而 `jiuding_dodsim` 已经支持从外部拓扑文件和流量文件读取输入。

下一步目标是让 `madrona_simple_example` 也支持：

- 从 Python 侧读取拓扑数据和流量数据
- 将这些结构化数据传入 Madrona
- 在 GPU/world 中被 `Sim` 正确读取
- 最终替代当前硬编码 topo / flow 的逻辑

本方案以 **输出等价优先** 为原则，优先保证 Madrona 与 jiuding 在相同输入下行为一致，再考虑后续清理与扩展。

---

## 2. 结论摘要

### 推荐方案

对于 **拓扑数据** 和 **流量数据**，推荐采用：

**Python 解析文件 → bindings.cpp 接收数组 → mgr.cpp 分配/拷贝共享输入块 → WorldInit 传入 Sim → loadTopo/loadFlow 从共享输入读取**

即复用当前仓库里已经存在的 **初始化期共享数据路径**。

### 不推荐的主方案

不建议优先把 topo / flow 做成：

- ECS component columns
- `exportColumn + to_torch()` 的 exported tensor

因为 topo / flow 的性质更接近：

- 初始化时一次性上传
- 所有 world 共享读取
- 并不是某个 archetype 的自然组件列

### exportColumn 仍然适用的范围

`exportColumn + to_torch()` 继续保留给：

- `Action`
- `Reset`
- 未来需要 **Python 每 step 改写、GPU 每 step 读取** 的动态字段

---

## 3. 现有仓库中已经存在的两条数据路径

## 3.1 动态输入路径：Action 的传递方式

当前仓库里，`Action` 已经是一条完整可工作的“Python → GPU 组件列”的样例。

涉及文件：

- `madrona_simple_example/src/types.hpp`
- `madrona_simple_example/src/sim.cpp`
- `madrona_simple_example/src/mgr.cpp`
- `madrona_simple_example/src/bindings.cpp`
- `madrona_simple_example/src/madrona_simple_example/gridworld.py`

数据路径为：

```text
Python 写 torch tensor
→ exported ECS column
→ GPU 上的 Action 组件列
→ tick / system 直接读取 Action
```

这条路径适合：

- 每 step 变化
- 按 world / agent 排列
- 天然属于组件列的数据

例如：

- action
- reset

---

## 3.2 初始化共享输入路径：GridState 的传递方式

当前仓库里还有另一条更适合 topo / flow 的数据路径，即 `GridState`。

涉及文件：

- `madrona_simple_example/src/grid.hpp`
- `madrona_simple_example/src/bindings.cpp`
- `madrona_simple_example/src/mgr.cpp`
- `madrona_simple_example/src/init.hpp`
- `madrona_simple_example/src/sim.cpp`

数据路径为：

```text
Python / NumPy
→ bindings.cpp 整理 host 数据
→ mgr.cpp 分配 GPU 数据块并 cudaMemcpy
→ WorldInit 把指针传给 Sim
→ Sim / systems 直接读取
```

这条路径更适合：

- 拓扑
- 流量定义
- 路由输入表
- 链路属性
- 其他初始化后长期不变的共享数据

因此，topo / flow 推荐走这条路，而不是 Action 那条路。

---

## 4. 为什么 topo / flow 不建议先做成组件数组

用户当前的直觉是：

> 把 topo 数据和 flow 数据读到组件数组里，然后在 init 以后解析出来，替代硬编码代码。

这个想法并不是完全错误，但不建议作为主方案，原因如下：

### 4.1 topo / flow 是全局输入，不是局部状态

拓扑和流量更像：

- 节点表
- 链路表
- 流定义表
- 路由构建输入

它们不是某个 entity 的自然“局部状态列”。

### 4.2 topo / flow 多数时候初始化一次即可

这些数据通常在：

- simulator 构造时
- episode 开始前

上传一次即可，后续主要是读，不会每 step 都由 Python 动态改写。

### 4.3 exportColumn 会增加不必要复杂度

如果强行用 exported tensor 来承载 topo / flow，会带来额外复杂度：

- slot 管理
- shape 管理
- tensor 生命周期管理
- 与 archetype/component 结构耦合

而对静态输入并没有明显收益。

---

## 5. 推荐的总体架构

## 5.1 设计原则

本次改造遵循以下原则：

1. **保留 jiuding 输入语义**
2. **保留现有 Madrona 核心 system 行为**
3. **尽量复用现有 GridState 路径**
4. **先做最小闭环，再做真实文件接入**
5. **优先保证 parity，不进行额外架构重构**

---

## 5.2 推荐的数据流

推荐最终采用的数据流如下：

```text
拓扑文件 / 流量文件
→ Python 解析
→ 结构化 numpy 数组
→ bindings.cpp 转成 NodeDef[] / LinkDef[] / FlowDef[]
→ mgr.cpp 创建共享 CPU/GPU 输入块
→ WorldInit 把共享输入指针传入 Sim
→ Sim::loadTopo() / Sim::loadFlow() 从共享输入读取
→ 替代 buildHardcodedTopo() / buildHardcodedFlows()
```

---

## 6. 建议复用的现有结构

## 6.1 输入结构直接复用

`madrona_simple_example/src/init.hpp` 中已经定义了：

- `NodeDef`
- `LinkDef`
- `FlowDef`

这些结构与 `jiuding_dodsim` 的输入语义基本一致，因此建议直接复用，不新增无必要抽象。

建议新增一个共享输入结构，例如：

```cpp
struct NetworkInit {
    const NodeDef *nodes;
    int32_t numNodes;

    const LinkDef *links;
    int32_t numLinks;

    const FlowDef *flows;
    int32_t numFlows;
};
```

然后扩展 `WorldInit`：

```cpp
struct WorldInit {
    EpisodeManager *episodeMgr;
    const GridState *grid;
    const NetworkInit *network;
};
```

这样可以把 topo / flow 作为一份 simulator-global shared input 交给所有 worlds。

---

## 7. 各文件建议改动

## 7.1 `madrona_simple_example/src/init.hpp`

### 目标

- 保留 `NodeDef` / `LinkDef` / `FlowDef`
- 新增 `NetworkInit`
- 扩展 `WorldInit`

### 作用

作为 Python 输入进入 Sim 的统一只读共享结构。

---

## 7.2 `madrona_simple_example/src/mgr.hpp`

### 目标

更新 `Manager` 构造函数签名，使其接收：

- `GridState`
- `NetworkInit`

例如概念上变成：

```cpp
Manager(const Config &cfg,
        const GridState &src_grid,
        const NetworkInit &src_network);
```

### 作用

让 Manager 同时负责：

- grid 初始化数据
- topo / flow 初始化数据

---

## 7.3 `madrona_simple_example/src/mgr.cpp`

### 目标

参考 `GridState` 的现有逻辑，为 `NetworkInit` 增加：

- CPU 模式 staging
- GPU 模式 staging
- 分配
- 拷贝
- 生命周期管理

### 具体建议

#### CPU 模式

- 分配 host 侧连续内存或独立数组
- 保存 `NodeDef[]` / `LinkDef[]` / `FlowDef[]` 副本
- 构造 `NetworkInit` 指向这批数据

#### CUDA 模式

- 像 `GridState` 一样手动：
  - `allocGPU(...)`
  - 构造 staging struct
  - `cudaMemcpy(...)`
- 使 GPU 侧 `NetworkInit` 内部指针指向 GPU 上的数组区域

#### WorldInit 传递

- 更新 `setupWorldInitData()`
- 将 `network` 指针传给每个 world

### 作用

把 Python 传进来的 topology / flow 数组真正变成 world 可读取的共享 GPU 数据块。

---

## 7.4 `madrona_simple_example/src/bindings.cpp`

### 目标

为 Python 接口增加 topology / flow 输入参数。

### 推荐方式

让 Python 传入标准 `numpy.ndarray`：

- nodes
- links
- flows

bindings 层负责：

- shape 检查
- dtype 检查
- 转为 `NodeDef[] / LinkDef[] / FlowDef[]`

### 作用

将 Python 侧的结构化实验输入交给 native 层，而不是让 native 层直接解析文本文件。

---

## 7.5 `madrona_simple_example/src/madrona_simple_example/gridworld.py`

### 目标

更新 Python API，使它除了当前 gridworld 参数外，也能接收：

- topology arrays
- flow arrays

### 建议

在第一阶段可保留一个 helper，用 Python 生成与当前硬编码场景完全等价的数据，作为迁移过渡。

### 作用

先验证“输入来源替换”不会导致行为变化，再接真实文件。

---

## 7.6 `madrona_simple_example/src/sim.hpp`

### 目标

必要时为 `Sim` 增加：

- `const NetworkInit *network;`

### 作用

保存来自 `WorldInit` 的共享输入指针，供 `loadTopo()` / `loadFlow()` 使用。

---

## 7.7 `madrona_simple_example/src/sim.cpp`

### 目标

在 `Sim` 构造函数里保存 `init.network`。

### 作用

把共享输入带入 world runtime。

---

## 7.8 `madrona_simple_example/src/sim_init.cpp`

### 目标

这是最核心的行为改造点。

把：

- `buildHardcodedTopo()`
- `buildHardcodedFlows()`

从主路径中移除，改为：

- `loadTopo()` 从 `network->nodes` / `network->links` 构建内部状态
- `loadFlow()` 从 `network->flows` 构建内部状态

### 必须保留的现有语义

- 链路双向展开
- `link.bandwidth > 0 ? link.bandwidth : node.port_bw`
- `pendingFlows` 按 `start_time` 排序
- `computeRoutes()` 逻辑不改
- 现有 port / peerPort / routeTable 逻辑不改

### 作用

真正完成从“硬编码输入”到“外部输入”的切换。

---

## 8. 与 jiuding 保持一致时必须注意的语义

以下语义必须保留，否则 parity 很容易失效。

## 8.1 `FlowDef.id` 不能变

`flow_id` 不只是一个标签，它会影响：

- ECMP 路径选择
- 流级行为一致性
- 最终完成时间输出

所以 Python 传入的 `flow_id` 必须原样保留，不要重编号。

---

## 8.2 默认值语义必须对齐 jiuding

需要保持：

- `NodeDef.port_bw` 默认 `100.0`
- `LinkDef.bandwidth == 0` 表示使用源节点 port bandwidth
- `FlowDef.priority` 默认 `0`

---

## 8.3 `start_time` 排序逻辑必须保留

当前 `loadFlow()` 会对 `pendingFlows` 按 `start_time` 排序。

这部分不要顺手改掉，否则调度顺序可能变化。

---

## 8.4 不要借机改动系统执行顺序

本次目标只是替换输入来源，不应同时改动：

- `schedulePendingFlows()`
- `deliverEvents()`
- `flowArrivalSystem()`
- `bwUpdateIngressSystem()`
- `portBandwidthAllocSystem()`
- `bufferUpdateSystem()`
- `flowProgressAndCleanupSystem()`

执行顺序一旦变化，输出 parity 很容易丢失。

---

## 9. 推荐迁移顺序

建议按以下顺序推进。

### 第一步：定义共享输入结构

- 新增 `NetworkInit`
- 扩展 `WorldInit`

### 第二步：打通 Manager 输入链路

- `bindings.cpp -> Manager -> mgr.cpp -> WorldInit`

### 第三步：先用 Python 构造“硬编码等价输入”

不要一上来就直接读取真实文件。

先在 Python 里生成与当前 `buildHardcodedTopo()` / `buildHardcodedFlows()` 完全一致的数据，验证：

- 新输入路径是通的
- 输出没有变化

### 第四步：切换 `loadTopo()` / `loadFlow()`

让 `Sim` 从 `NetworkInit` 读取，而不是再调用硬编码 builder。

### 第五步：接入真实文件读取

等“硬编码等价数据”验证通过后，再在 Python 侧接：

- topo 文件解析
- flow 文件解析

### 第六步：做 parity 验证

用 jiuding 真实示例输入做端到端对比。

---

## 10. 验证方案

## 10.1 第一阶段：只验证输入通路替换

目标：

- Python 输入 == 当前硬编码输入
- Madrona 行为不变

建议检查：

- init log
- `numTopoNodes`
- `numTopoLinks`
- `numPorts`
- 邻居关系
- route 结果
- `numFlowDefs`
- `pendingFlows` 顺序

---

## 10.2 第二阶段：接真实 jiuding 示例

建议直接使用现有示例：

- `jiuding_dodsim/examples/leafspine64/leafspine_h64_topo.txt`
- `jiuding_dodsim/examples/leafspine64/leafspine_h64_d8_alltoall_2mb.txt`

比较内容：

- init parity
- 系统阶段日志 parity
- flow completion / FCT parity

---

## 10.3 建议复用的已有验证资源

可复用当前工作区已有脚本和产物：

- `run_jiuding_init_log.sh`
- `run_madrona_init_log.sh`
- 工作区现有 `jiuding_*log`
- 工作区现有 `madrona_*log`
- `out/flow_completion_times.csv`

---

## 11. 风险与约束

## 11.1 固定上限暂时不要重构

当前 `Sim` 中有固定上限，例如：

- `MAX_TOPO_NODES`
- `MAX_TOPO_LINKS`
- `MAX_FLOWS`

第一阶段先不要重构这些上限。

推荐做法：

- 在 binding 或 manager 边界检查输入数量
- 超限则明确报错

这样风险最低。

---

## 11.2 先做最小闭环，不要一次到位

不建议第一步就同时做：

- 文件解析
- Python API 改造
- GPU 共享输入块
- Sim 逻辑替换
- parity 对比

建议先做“硬编码等价输入”的最小闭环，再逐步扩展。

---

## 12. 最终建议

对于“让 Madrona 支持从外部读取拓扑与流量数据，而不是硬编码”这个目标，推荐采用：

**Python 解析文件 + 共享输入块上传 + WorldInit 传给 Sim**

而不是把 topo / flow 优先做成组件数组或 exported tensors。

这是当前仓库里：

- 与数据性质最匹配
- 可复用现有代码最多
- 对输出等价风险最小

的实现方案。

如果后续只有某些小字段需要被 Python 在每 step 动态修改，再把那部分单独拆到 `exportColumn + to_torch()` 路径即可。
