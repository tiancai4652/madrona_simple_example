# `sim.cpp` 拆分方案

## 1. 目标

当前 `madrona_simple_example/src/sim.cpp` 已经同时承载了：

- world 注册与任务图组装
- 初始化与硬编码场景装载
- 拓扑 / 路由 / 路径查询
- 事件队列与延迟传播
- flow/tag 生命周期
- 带宽分配
- buffer 推进
- PFC 逻辑
- init 对比日志

这会带来两个直接问题：

1. 文件过长，阅读和定位成本高
2. 运行阶段逻辑和初始化逻辑混在一起，不利于后续继续做 Jiuding / Madrona 对齐

本次拆分目标不是“做架构重写”，而是：

- **在不改变业务语义的前提下拆文件**
- **保留 Madrona 风格的清晰任务图入口**
- **同时让文件分层能和 Jiuding 源码形成对照**

也就是说，拆分后仍然要满足：

- `Sim` 仍然是唯一 world 业务中心
- `setupTasks(...)` 仍然是一眼能看懂主循环顺序的地方
- 组件 / archetype / world 数据的边界不变
- 不引入额外的“面向对象包装层”或复杂抽象

---

## 2. 拆分时必须遵守的原则

这些原则直接来自 `madrona_simple_example/MADRONA_BUSINESS_GUIDE.md`，也是本方案的约束。

### 2.1 状态定义不因为拆文件而分散

按照 guide：

- `src/types.hpp`：组件、业务枚举、archetype
- `src/init.hpp` / `src/grid.hpp`：world 初始化输入与共享只读数据
- `src/sim.hpp`：`Sim` world 类型、配置、上下文、world 级业务状态

因此本次**不建议**把这些业务状态拆成很多新的 header。

拆分的重点应该放在：

- `Sim::method` 的定义位置
- 业务阶段如何分文件

而不是把状态定义打散。

### 2.2 按“业务阶段”拆，不按“语法技巧”拆

guide 明确建议：

- 系统函数应当对应**明确业务阶段**
- 复杂业务优先拆成多个短系统，而不是一个巨大的总系统
- 任务图顺序要能体现业务阶段顺序

所以这次拆分应优先围绕：

- init
- route / path / world helper
- event / arrival / cleanup
- bandwidth
- buffer
- pfc
- debug log

而不是按“公共函数 / 工具函数 / util / helper”这种泛化分类去拆。

### 2.3 `setupTasks(...)` 必须保留在主入口文件

`Sim::setupTasks(...)` 现在位于 `madrona_simple_example/src/sim.cpp:404`，它直接展示了运行阶段顺序：

1. schedule pending flows
2. deliver events
3. arrival
4. bw update
5. pfc propagate
6. port alloc
7. pfc detect
8. downstream emit
9. clear dirty
10. choose dt
11. buffer update
12. flow progress / cleanup

这部分是理解业务主循环的核心视图。

**建议保留在 `sim.cpp`**，不要挪到隐藏很深的文件里。

### 2.4 不引入额外“服务类 / 管理器类”

guide 强调业务代码要保持：

- POD
- 定长数组
- 显式控制流
- 明确的 world 状态

所以本次拆分应当是：

- 仍然用 `Sim::xxx()` 成员函数
- 只是把定义分散到多个 `.cpp`

而**不建议**引入：

- `RouteManager`
- `BufferService`
- `PfcController`
- `FlowRuntimeHelper`

这类额外对象层。

### 2.5 共享私有辅助逻辑优先做成 `Sim` 私有成员

目前 `sim.cpp` 里有一部分匿名 namespace 内的辅助函数，例如 init log 相关逻辑位于 `madrona_simple_example/src/sim.cpp:18` 起。

拆分后如果某个辅助逻辑要跨文件被 `Sim` 的其他方法调用，优先做法是：

- 把它改成 `Sim` 的私有成员函数，或
- 保持在单个 `.cpp` 内部使用

而不是一开始就建立很大的内部 header 工具层。

---

## 3. 当前 `sim.cpp` 的自然业务分块

根据当前代码，`sim.cpp` 已经天然分成若干块：

| 当前区块 | 参考位置 | 说明 |
|---|---:|---|
| init log / 调试输出 | `sim.cpp:18`、`sim.cpp:49`、`sim.cpp:142` | 仅用于 init 对比，且 CPU-only |
| task node 封装 | `sim.cpp:204` 起 | 把每个业务阶段挂到 task graph |
| register / setupTasks | `sim.cpp:375`、`sim.cpp:404` | Madrona 世界入口 |
| reset / world 清零 | `sim.cpp:422` | world 状态重置 |
| 硬编码 topo / flow | `sim.cpp:579`、`sim.cpp:632` | 当前 leafspine64 初始化数据 |
| routing / path / route helper | `sim.cpp:662` 起 | 路由、ECMP、路径查询 |
| timer helper | `sim.cpp:887` 起 | backlog / pfc timer 维护 |
| tag / event / completion helper | `sim.cpp:979`、`sim.cpp:1030`、`sim.cpp:1119` | flow tag 生命周期、延迟事件 |
| inject / schedule / deliver / arrival / bw update | `sim.cpp:1352` 到 `sim.cpp:1580` | 更接近 Jiuding `systems.cpp` |
| port bandwidth alloc | `sim.cpp:1605` | 更接近 Jiuding `systems_bandwidth.cpp` |
| pfc detect / propagate | `sim.cpp:1580`、`sim.cpp:2123` | 更接近 Jiuding `systems_pfc.cpp` |
| downstream emit | `sim.cpp:2411` | 带宽分配后向下游发事件 |
| buffer update | `sim.cpp:2540` | 更接近 Jiuding `systems_buffer.cpp` |
| flow progress / cleanup | `sim.cpp:2901` | 运行尾段推进 |
| loadTopo / loadFlow / ctor | `sim.cpp:3165`、`sim.cpp:3253`、`sim.cpp:3284` | 世界初始化主路径 |

这说明：当前代码虽然都堆在一个文件里，但业务边界其实已经比较清楚。

因此拆分不需要重设计，只需要把已有边界显式化。

---

## 4. 推荐方案：Madrona 主入口保留 + Jiuding 对齐拆分

我建议采用**混合式拆分**：

- **入口层**保持 Madrona 风格
- **大块业务逻辑**按 Jiuding 的系统分类拆出去

这样既符合 guide，也方便后面对照 `jiuding_dodsim/src/*.cpp`。

### 4.1 拆分后的目标文件结构

建议拆成下面这些文件：

```text
src/
  sim.hpp
  sim.cpp
  sim_init.cpp
  sim_world.cpp
  sim_debug.cpp
  sim_systems.cpp
  sim_systems_bandwidth.cpp
  sim_systems_buffer.cpp
  sim_systems_pfc.cpp
```

下面是每个文件的职责。

---

## 5. 每个目标文件放什么

### 5.1 `src/sim.cpp` —— 保留主入口和任务图

**职责：**

- task node 定义
- `Sim::registerTypes(...)`
- `Sim::setupTasks(...)`
- `Sim::Sim(...)`

**为什么保留这里：**

这是 Madrona 业务代码的“总入口”。

从 guide 看，复杂业务也应该保证：

- world 创建入口集中可见
- task graph 顺序集中可见

所以这部分不建议继续拆散。

**当前对应代码：**

- task nodes：`madrona_simple_example/src/sim.cpp:204`
- `registerTypes(...)`：`madrona_simple_example/src/sim.cpp:375`
- `setupTasks(...)`：`madrona_simple_example/src/sim.cpp:404`
- 构造函数：`madrona_simple_example/src/sim.cpp:3284`

**备注：**

构造函数里只保留高层流程，例如：

- `resetNetworkState()`
- 读取 cfg
- `loadTopo(ctx)`
- `loadFlow(ctx)`

这样初始化路径仍然一眼可见。

---

### 5.2 `src/sim_init.cpp` —— 初始化与硬编码场景装载

**职责：**

- `Sim::resetNetworkState()`
- `Sim::createPort(...)`
- `Sim::buildHardcodedTopo(...)`
- `Sim::buildHardcodedFlows(...)`
- `Sim::loadTopo(...)`
- `Sim::loadFlow(...)`

**为什么放一起：**

这些逻辑都属于“世界如何被构出来”。

而且 guide 强调：

- 初始化逻辑应集中
- 世界创建要一处可见

当前 `loadTopo(...)` 与 `loadFlow(...)` 是初始化边界，和 leafspine64 硬编码输入天然属于一组。

**当前对应代码：**

- `resetNetworkState()`：`madrona_simple_example/src/sim.cpp:422`
- `buildHardcodedTopo(...)`：`madrona_simple_example/src/sim.cpp:579`
- `buildHardcodedFlows(...)`：`madrona_simple_example/src/sim.cpp:632`
- `loadTopo(...)`：`madrona_simple_example/src/sim.cpp:3165`
- `loadFlow(...)`：`madrona_simple_example/src/sim.cpp:3253`

**和 Jiuding 的关系：**

这部分职责大致对应：

- `jiuding_dodsim/src/world.cpp:24` 的端口创建
- `jiuding_dodsim/src/world.cpp:55` 的 `load_topology(...)`
- `jiuding_dodsim/src/world.cpp:164` 的 `load_flows(...)`

只是 Madrona 当前还带有“硬编码场景生成”这一层，所以单独拉出 `sim_init.cpp` 更合适。

---

### 5.3 `src/sim_world.cpp` —— world 级辅助逻辑、路由、路径、时间推进决策

**职责：**

- `Sim::findNodeSlot(...)`
- `Sim::findNeighborSlot(...)`
- `Sim::computeRoutes()`
- `Sim::getPath(...)`
- `Sim::lookupFlowRouteNext(...)`
- `Sim::pushDelayedEvent(...)`
- `Sim::chooseDT() const`
- `Sim::computePropagationTimeAt(...) const`
- `Sim::computePropagationTime(...) const`
- `Sim::computePropagationTimeForPort(...) const`

**为什么放一起：**

这些都不是某个单独系统阶段，而是整个 world 运行过程中反复依赖的“世界级公共业务能力”：

- 路由查询
- 延迟传播时间计算
- 未来事件插入
- 下一步 `dt` 选择

这类逻辑在 Jiuding 里主要散落在 `world.cpp`，因此这里命名成 `sim_world.cpp` 最容易对齐用户的源代码心智。

**当前对应代码：**

- `computeRoutes()`：`madrona_simple_example/src/sim.cpp:662`
- timer / propagation / queue / dt 相关逻辑分布在 `sim.cpp:887`、`sim.cpp:1119` 以及后续若干段

**和 Jiuding 的关系：**

最接近：

- `jiuding_dodsim/src/world.cpp:185` 的 `compute_routes()`
- `jiuding_dodsim/src/world.cpp` 中其它 world 级辅助逻辑

---

### 5.4 `src/sim_debug.cpp` —— init 对比日志

**职责：**

- `init_log_print_enabled`
- init log 相关 helper
- topo init log
- flow init log

**建议做法：**

把现在匿名 namespace 的：

- `printInitTopoLog(...)`
- `printInitFlowLog(...)`

改成更稳定的内部组织方式。推荐两种做法中选一种：

1. 改成 `Sim` 私有成员函数，例如：
   - `Sim::printInitTopoLog(Engine &ctx) const`
   - `Sim::printInitFlowLog() const`
2. 或者在 `sim_debug.cpp` 内部保留 file-local helper，再通过少量对外声明暴露调用入口

**我更推荐第 1 种。**

原因：

- 避免再加一个内部 debug header
- 调用点仍然清楚
- debug 输出本来就是从 `Sim` 状态读取

**当前对应代码：**

- `madrona_simple_example/src/sim.cpp:18`
- `madrona_simple_example/src/sim.cpp:49`
- `madrona_simple_example/src/sim.cpp:142`

**为什么单独拆：**

这部分是 CPU-only 的调试/比对逻辑，不属于核心业务推进。继续放在主文件里会污染核心阅读路径。

---

### 5.5 `src/sim_systems.cpp` —— 对齐 Jiuding `systems.cpp`

**职责：**

建议放这几类逻辑：

- `Sim::injectFlow(...)`
- `Sim::injectFlowDef(...)`
- `Sim::schedulePendingFlows()`
- `Sim::deliverEvents()`
- `Sim::flowArrivalSystem(Engine &ctx)`
- `Sim::bwUpdateIngressSystem(Engine &ctx)`
- `Sim::findTag(...) const`
- `Sim::createTagOnPort(...)`
- `Sim::destroyTag(...)`
- `Sim::recordFlowCompletion(...)`
- `Sim::clearDirtyPorts(Engine &ctx)`
- `Sim::flowProgressAndCleanupSystem(Engine &ctx, Time dt)`

**为什么这样分：**

这些函数都围绕：

- 事件进入 world
- tag 被创建 / 更新 / 销毁
- source / flow 生命周期推进
- 每轮结尾 cleanup

它们最接近 Jiuding 里的“基础系统层”。

**当前对应代码：**

- `injectFlow(...)`：`madrona_simple_example/src/sim.cpp:1352`
- `injectFlowDef(...)`：`madrona_simple_example/src/sim.cpp:1374`
- `schedulePendingFlows()`：`madrona_simple_example/src/sim.cpp:1432`
- `deliverEvents()`：`madrona_simple_example/src/sim.cpp:1444`
- `flowArrivalSystem(...)`：`madrona_simple_example/src/sim.cpp:1473`
- `bwUpdateIngressSystem(...)`：`madrona_simple_example/src/sim.cpp:1503`
- `clearDirtyPorts(...)`：`madrona_simple_example/src/sim.cpp:2465`
- `flowProgressAndCleanupSystem(...)`：`madrona_simple_example/src/sim.cpp:2901`

**和 Jiuding 的关系：**

最接近：

- `jiuding_dodsim/src/systems.cpp`

---

### 5.6 `src/sim_systems_bandwidth.cpp` —— 对齐 Jiuding `systems_bandwidth.cpp`

**职责：**

- `Sim::portBandwidthAllocSystem(Engine &ctx, Time dt)`
- `Sim::downstreamEmitSystem(Engine &ctx)`

**为什么单独拆：**

`portBandwidthAllocSystem(...)` 是当前最重的一段业务逻辑之一，单体过大，且业务含义明确：

- 脏端口竞争
- 不同 QoS 模式分配
- 计算 `out_bw`
- 后续向下游发送事件

把它和其它逻辑拆开，收益非常高。

**当前对应代码：**

- `portBandwidthAllocSystem(...)`：`madrona_simple_example/src/sim.cpp:1605`
- `downstreamEmitSystem(...)`：`madrona_simple_example/src/sim.cpp:2411`

**和 Jiuding 的关系：**

最接近：

- `jiuding_dodsim/src/systems_bandwidth.cpp`

---

### 5.7 `src/sim_systems_buffer.cpp` —— 对齐 Jiuding `systems_buffer.cpp`

**职责：**

- `Sim::materializeBacklog(...)`
- `Sim::materializeRemaining(...)`
- `Sim::alignChunksWithBufCnt(...)`
- `Sim::materializeBufCnt(...)`
- `Sim::drainBufferChunks(...)`
- `Sim::bufferUpdateSystem(Engine &ctx, Time dt)`

**为什么单独拆：**

buffer / backlog / draining 是这个业务里最容易改坏语义的部分之一。

而 Jiuding 侧也已经把它独立成：

- `src/systems_buffer.cpp`

因此 Madrona 这边也应当把 buffer 逻辑独立出来，避免和带宽分配、cleanup、PFC 交叉在一个大文件里。

**当前对应代码：**

- buffer materialize helper：`madrona_simple_example/src/sim.cpp:979`、`madrona_simple_example/src/sim.cpp:1170`、`madrona_simple_example/src/sim.cpp:1214`、`madrona_simple_example/src/sim.cpp:1252`
- `bufferUpdateSystem(...)`：`madrona_simple_example/src/sim.cpp:2540`

**和 Jiuding 的关系：**

最接近：

- `jiuding_dodsim/src/systems_buffer.cpp`

---

### 5.8 `src/sim_systems_pfc.cpp` —— 对齐 Jiuding `systems_pfc.cpp`

**职责：**

- `Sim::setBacklogDrainTimer(...)`
- `Sim::setPfcPauseTimer(...)`
- `Sim::setPfcResumeTimer(...)`
- `Sim::clearBacklogDrainTimer(...)`
- `Sim::clearPfcPauseTimer(...)`
- `Sim::clearPfcResumeTimer(...)`
- `Sim::pfcPropagateSystem(Engine &ctx)`
- `Sim::pfcThresholdDetectSystem(Engine &ctx)`

**为什么这样分：**

这些逻辑围绕的是同一件事：

- PFC 控制状态
- 与 PFC 直接相关的 timer 生命周期

虽然 `backlogDrainTimer` 严格说不完全等于 PFC，但它和 pause / resume timer 都属于“基于未来时间点触发”的端口控制计时器，放一起更利于后面统一看 `chooseDT()` 依赖链。

**当前对应代码：**

- timer helper：`madrona_simple_example/src/sim.cpp:887` 到 `madrona_simple_example/src/sim.cpp:955`
- `pfcPropagateSystem(...)`：`madrona_simple_example/src/sim.cpp:1580`
- `pfcThresholdDetectSystem(...)`：`madrona_simple_example/src/sim.cpp:2123`

**和 Jiuding 的关系：**

最接近：

- `jiuding_dodsim/src/systems_pfc.cpp`

---

## 6. 为什么不建议按“实体”来拆

一种常见直觉是拆成：

- `port.cpp`
- `flowtag.cpp`
- `route.cpp`

但这个项目不太适合这样拆。

原因是：

1. **当前业务推进单位是“系统阶段”，不是“实体类方法”**
2. 同一个阶段会同时读写：
   - `PortState`
   - `FlowTagState`
   - `PortBuffer`
   - `PortPfcState`
3. 如果按实体拆，`port alloc`、`buffer update`、`cleanup`、`pfc` 会反复跨文件跳转，反而更难看

而 guide 也强调：

- 复杂业务应当按系统阶段拆
- 任务图顺序应直接对应业务阶段

所以这次更适合按**系统阶段 / world 职责**拆，而不是按实体拆。

---

## 7. 头文件层面的建议

### 7.1 `sim.hpp` 先保留为统一声明头

本轮只做 `.cpp` 拆分时，建议 `sim.hpp` 继续承担：

- `Sim` 成员声明
- world 状态数组
- 常量上限
- world 级辅助结构体

这样改动最小，最安全。

### 7.2 如果后续 `sim.hpp` 也变大，再考虑第二阶段 header 拆分

但这一步**不建议现在就做**。

原因：

- 当前主要问题是 `sim.cpp` 太长
- 如果第一轮就同时改很多 header，会增加编译错误和语义漂移风险

如果以后要继续拆 header，我建议按下面顺序：

1. 先拆 `.cpp`
2. 再观察 `sim.hpp` 是否真的难维护
3. 只有在确认必要时，再引入例如：
   - `sim_state.hpp`
   - `sim_events.hpp`
   - `sim_route.hpp`

否则先不要动。

---

## 8. CMake 需要怎样改

当前 `madrona_simple_example/src/CMakeLists.txt:1` 里：

```cmake
set(SIMULATOR_SRCS
    types.hpp sim.hpp sim.cpp
)
```

拆分后建议改成：

```cmake
set(SIMULATOR_SRCS
    types.hpp
    sim.hpp
    sim.cpp
    sim_init.cpp
    sim_world.cpp
    sim_debug.cpp
    sim_systems.cpp
    sim_systems_bandwidth.cpp
    sim_systems_buffer.cpp
    sim_systems_pfc.cpp
)
```

这样 CPU / GPU 两条构建路径都会自动使用新的源文件集合，因为当前 `src/CMakeLists.txt:5` 和 `src/CMakeLists.txt:28` 都依赖 `SIMULATOR_SRCS`。

---

## 9. 推荐实施顺序

建议按“只搬文件、不改语义”的顺序做。

### 第 1 步：先拆最独立的块

优先拆：

- `sim_debug.cpp`
- `sim_systems_bandwidth.cpp`
- `sim_systems_buffer.cpp`
- `sim_systems_pfc.cpp`

原因：

- 这些块业务边界最清楚
- 拆出后 `sim.cpp` 会立刻瘦很多
- 语义迁移相对可控

### 第 2 步：拆初始化与 world 级逻辑

再拆：

- `sim_init.cpp`
- `sim_world.cpp`

原因：

- 这一步会涉及更多 helper 之间的相互调用
- 适合在大系统块拆完之后再做

### 第 3 步：最后保留 `sim.cpp` 为薄入口

最终让 `sim.cpp` 只保留：

- include
- task nodes
- `registerTypes(...)`
- `setupTasks(...)`
- `Sim::Sim(...)`

这样以后查看主循环时，第一眼就能看到入口，不会被大量系统细节淹没。

---

## 10. 每一步拆完后的校验方式

因为当前项目最重要的是和 Jiuding 保持语义一致，所以拆分时建议每一步都做**等价性校验**。

### 10.1 最低限度校验

每拆完一个文件后：

1. `cmake --build ./madrona_simple_example/build -j$(nproc)`
2. 运行 `./run_madrona_init_log.sh`
3. 确认 init 日志没有变化

### 10.2 阶段性校验

当全部拆完后：

1. 再跑一遍 Jiuding / Madrona init log 对比
2. 确认 `diff -u` 仍为空
3. 然后再恢复运行阶段的下一轮对齐

这样可以保证这次拆分只是“代码组织调整”，不是业务语义修改。

---

## 11. 我不建议本轮做的事情

### 11.1 不建议同时重命名大量业务函数

例如：

- `flowArrivalSystem(...)`
- `portBandwidthAllocSystem(...)`
- `bufferUpdateSystem(...)`

这些名字已经直接对应业务阶段，且能和 Jiuding 心智对齐。

本轮拆分时最好**只搬位置，不改命名**。

### 11.2 不建议把系统改成很多泛化 helper

例如不要为了“复用”而拆成：

- `updateFlowTagCommon(...)`
- `applyPortMutation(...)`
- `dispatchEventGeneric(...)`

这会降低业务可读性，也不符合 guide 强调的“平铺、直接、显式”的风格。

### 11.3 不建议把 `setupTasks(...)` 拆散

任务图顺序是排查 parity 问题时最重要的入口。

如果把顺序关系藏进多个文件，会显著增加调试成本。

---

## 12. 最终建议

如果只给一个结论，我建议采用下面这套拆法：

### 推荐拆法

- `sim.cpp`：Madrona 入口层
- `sim_init.cpp`：初始化与场景装载
- `sim_world.cpp`：路由 / world helper / chooseDT
- `sim_debug.cpp`：init 对比日志
- `sim_systems.cpp`：事件 / arrival / cleanup / flow 生命周期
- `sim_systems_bandwidth.cpp`：带宽分配与下游发射
- `sim_systems_buffer.cpp`：buffer / backlog / draining
- `sim_systems_pfc.cpp`：PFC 与相关 timer

### 这套方案的好处

1. **符合 Madrona guide**
   - 主入口、任务图、world 结构仍清晰
2. **对齐 Jiuding 源码分层**
   - 后续对照 `systems.cpp / systems_bandwidth.cpp / systems_buffer.cpp / systems_pfc.cpp` 更直接
3. **风险可控**
   - 以“搬运定义位置”为主，不要求重写业务逻辑
4. **后续更容易继续做运行阶段对齐**
   - 主循环每个阶段会对应到更稳定的文件边界

---

## 13. 建议的第一轮落地顺序

如果下一步要真正开始拆，我建议按下面顺序动手：

1. `sim_debug.cpp`
2. `sim_systems_bandwidth.cpp`
3. `sim_systems_buffer.cpp`
4. `sim_systems_pfc.cpp`
5. `sim_systems.cpp`
6. `sim_init.cpp`
7. `sim_world.cpp`
8. 最后把 `sim.cpp` 收薄并整理 `src/CMakeLists.txt`

这样每一步都比较容易编译通过，也方便随时停下来做 parity 校验。
