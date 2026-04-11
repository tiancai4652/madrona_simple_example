# Madrona Python → GPU 数据传输说明

本文档用于指导其他 Agents 在本仓库中实现“从 Python 接口输入数据，并让 GPU 侧 Madrona 仿真代码可读取”的功能。

重点不是泛泛而谈，而是基于本仓库当前已经存在的两条真实数据路径来说明：

1. **构造期上传路径**：Python / NumPy → C++ bindings → `Manager` → GPU 内存中的结构体数据
2. **运行期读写路径**：Python / PyTorch → Madrona exported tensor → ECS 组件列（CPU / GPU）

这两条路径分别对应你提到的：
- 拓扑 / 流量这类“外部输入的大块静态或半静态数据”
- action / reset 这类“每步由 Python 写入、由仿真器读取”的字段

---

## 1. 先看现有的真实范例：Action 是怎么从 Python 传到 Agent 组件的

当前仓库里，`Agent::Action` 已经是一个完整可工作的样例。

### 1.1 组件定义

`Action` 组件定义在 `src/types.hpp:20`，并且属于 `Agent` archetype，见 `src/types.hpp:45`。

```cpp
struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep
> {};
```

这意味着每个 world 中的 agent 实体都拥有一列 `Action` 数据。

### 1.2 在 ECS 注册阶段把这一列导出给 Python

`src/sim.cpp:23`：

```cpp
registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
```

这一步很关键。

它的含义是：
- 把 `Agent` archetype 里的 `Action` 这一列暴露出去
- 并把它绑定到 `ExportID::Action`
- 后续 `Manager` 可以通过这个 slot 取回底层 buffer，并包装成 Python / Torch 可访问的 tensor

### 1.3 Manager 根据 ExportID 取回底层 buffer

`src/mgr.cpp:241`：

```cpp
Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {impl_->cfg.numWorlds, 1});
}
```

这里的 `actionTensor()` 并没有自己分配一块新内存。
它只是把 Madrona 内部已经为导出列准备好的那块内存，以 `madrona::py::Tensor` 的形式暴露给 Python。

而真正返回底层指针的地方在：
- CPU: `src/mgr.cpp:73`
- GPU: `src/mgr.cpp:119`

```cpp
void *dev_ptr = gpuExec.getExported((uint32_t)slot);
return Tensor(dev_ptr, type, dims, cfg.gpuID);
```

对 GPU 模式来说，这里的 `dev_ptr` 指向的是 **GPU 上的导出列内存**。
因此，Python 拿到的 Torch tensor 实际上是和 GPU 上这块 ECS 列数据共享 / 对接的。

### 1.4 Python 把这个 exported tensor 转成 torch tensor

`src/madrona_simple_example/gridworld.py:35`：

```python
self.actions = self.sim.action_tensor().to_torch()
```

所以 `self.actions` 不是普通 Python list，也不是额外复制出来的一份缓存。
它就是对 Madrona 导出列的 Torch 视图。

### 1.5 Python 侧写入 action，GPU 侧 tick 直接读取

`scripts/run.py:30`：

```python
grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
```

而在仿真 step 中，`src/sim.cpp:30` 的 `tick(...)` 直接读取 `Action &action`：

```cpp
inline void tick(Engine &ctx,
                 Action &action,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step)
```

也就是说，数据路径是：

```text
Python 写 self.actions
→ Torch tensor
→ Madrona exported Action column
→ GPU 上 Agent::Action 组件列
→ tick(Action &action, ...) 直接读取
```

这就是当前仓库里“从外部 Python 字段传到 Madrona 组件里”的标准模板。

---

## 2. 再看另一条真实范例：Grid / 拓扑类数据是怎么进 GPU 的

与 action 不同，`grid` 不是 exported ECS 列，而是 **world 共享的结构化只读数据**。

它当前的路径是：

```text
Python numpy 参数
→ bindings.cpp 组装 Cell / GridState
→ Manager::Impl::init(...)
→ CUDA 分支手动分配 GPU 内存
→ cudaMemcpy 拷入 GPU
→ WorldInit 把指针传给每个 world
→ Sim / tick 通过 ctx.data().grid 读取
```

### 2.1 Python 构造时把 walls / rewards / end_cells 传入 C++

`src/madrona_simple_example/gridworld.py:22-31` 里，Python 初始化 `SimpleGridworldSimulator(...)` 时传入：
- `walls`
- `rewards`
- `end_cells`
- `start_x / start_y`
- `num_worlds`
- `exec_mode`

### 2.2 bindings.cpp 把 Python 输入转换成原生 `Cell[]`

`src/bindings.cpp:56` 的 `setupCellData(...)` 会把：
- `walls`
- `rewards`
- `end_cells`

整理成一块 C++ `Cell *cells`。

然后在 `src/bindings.cpp:105` 构造：

```cpp
new (self) Manager(..., GridState {
    .cells = cells,
    .startX = ...,
    .startY = ...,
    .width = ...,
    .height = ...,
});
```

注意这里传给 `Manager` 的 `GridState` 仍然是 **CPU 侧暂存对象**。

### 2.3 Manager 的 CUDA 分支负责真正开辟 GPU 数据块

关键代码在 `src/mgr.cpp:179-216`。

先分配：
- `EpisodeManager` 的 GPU 空间：`cu::allocGPU(sizeof(EpisodeManager))`
- `GridState + Cell[]` 的 GPU 空间：`cu::allocGPU(sizeof(GridState) + num_cell_bytes)`

```cpp
auto *grid_data =
    (char *)cu::allocGPU(sizeof(GridState) + num_cell_bytes);
Cell *gpu_cell_data = (Cell *)(grid_data + sizeof(GridState));
```

这里就是你提到的“在 GPU 中开辟一块空间，并将其绑定到一个数据结构里”。

其做法是：
- 先申请一整块连续显存
- 前半部分解释成 `GridState`
- 后半部分解释成 `Cell[]`
- 再构造一个 host 侧 staging 结构，把 `GridState.cells` 指向 GPU 上的 `gpu_cell_data`

```cpp
GridState grid_staging {
    .cells = gpu_cell_data,
    .startX = src_grid.startX,
    .startY = src_grid.startY,
    .width = src_grid.width,
    .height = src_grid.height,
};
```

### 2.4 再把 host 数据拷到 GPU

`src/mgr.cpp:205-208`：

```cpp
cudaMemcpy(grid_data, &grid_staging, sizeof(GridState),
           cudaMemcpyHostToDevice);
cudaMemcpy(gpu_cell_data, src_grid.cells, num_cell_bytes,
           cudaMemcpyHostToDevice);
```

这一步完成后：
- GPU 上有一个 `GridState`
- 它的 `cells` 指针也指向 GPU 上的 `Cell[]`

### 2.5 通过 WorldInit 把这个 GPU 指针传到每个 world

`src/mgr.cpp:125` 的 `setupWorldInitData(...)` 会构造每个 world 的 `WorldInit`：

```cpp
world_inits[i] = WorldInit {
    episode_mgr,
    grid,
};
```

在 GPU 模式下，这里的 `grid` 已经是 `gpu_grid`，也就是 GPU 上那块 `GridState *`。

随后 `Sim` 构造时接收它，见 `src/sim.cpp:130-144`：

```cpp
Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
```

之后 `tick()` 中直接通过 `ctx.data().grid` 访问：

```cpp
const GridState *grid = ctx.data().grid;
```

因此这条路径说明：

- **不是所有数据都必须走 exportColumn**
- 对于拓扑、流量矩阵、链路属性这类“大块只读 / 共享场景数据”，更自然的方式通常是：
  - 在 `bindings.cpp` / `Manager` 初始化时接收 Python 输入
  - 在 `mgr.cpp` CUDA 分支手动分配 GPU 内存
  - 用 `WorldInit` 或 world data 把 GPU 指针传入仿真逻辑

---

## 3. 当前仓库里其实已经存在两种不同的数据接入模式

### 模式 A：导出列 / Torch 直连模式

适用于：
- action
- reset
- 未来每 step 会频繁由 Python 改写的数据
- shape 与 world / agent 数量强相关的数据

特点：
- 数据是 ECS component column 的一部分
- 通过 `exportColumn` 暴露
- 由 `Manager::exportTensor()` 返回 `madrona::py::Tensor`
- Python 通过 `.to_torch()` 得到 tensor
- Python 改写 tensor，仿真 step 直接读组件列

当前样例：
- `Action`：`src/sim.cpp:24`
- `Reset`：`src/sim.cpp:23`
- Python 侧句柄：`src/madrona_simple_example/gridworld.py:34-38`

### 模式 B：初始化上传 / 共享 GPU 结构体模式

适用于：
- 拓扑
- 链路容量
- 路由表
- 流量矩阵
- 其他大块、结构化、主要用于仿真器读取的数据

特点：
- 数据不是某个 entity 的 component column
- 而是 manager 手动分配的一块 GPU 数据结构
- 通过 `WorldInit` / `Sim` 保存指针
- `tick()` 或其他 system 直接解引用读取

当前样例：
- `GridState` / `Cell[]`，见 `src/mgr.cpp:193-208`

---

## 4. 给其他 Agents 的实现建议：拓扑 / 流量数据应该优先走哪条路

结合你现在的目标：
- jiuding 仿真器可以读取拓扑数据文件、流量数据文件
- madrona 目前把这些信息硬编码在代码里
- 下一步想让 madrona 也能读取外部数据

**建议优先采用“模式 B：初始化上传 / 共享 GPU 结构体模式”来处理拓扑和流量基础数据。**

原因：

1. **拓扑 / 流量通常不是按 agent 一列一列组织的**
   - 它们更像全局数组、矩阵、邻接表、边列表、节点属性表
   - 强行塞进 archetype component column 往往不自然

2. **这些数据往往在一次 episode 或一次构造期间基本稳定**
   - 更适合初始化时一次性上传到 GPU
   - 然后被多个 world / system 重复读取

3. **当前 `GridState` 已经提供了现成模板**
   - 直接类比扩展为 `TopologyState`、`TrafficState` 即可

如果以后某些流量字段需要在每一步由 Python 动态修改，再单独把那部分拆成 exported column / tensor 模式。

---

## 5. 推荐的实现蓝图

下面给其他 Agents 一个可执行的思路。

### 5.1 为拓扑 / 流量定义原生结构体

参考 `src/grid.hpp:10-22` 的做法，新增类似：

```cpp
struct LinkData {
    int32_t src;
    int32_t dst;
    float capacity;
};

struct TopologyState {
    const LinkData *links;
    int32_t numLinks;
    int32_t numNodes;
};

struct TrafficDemand {
    int32_t src;
    int32_t dst;
    float demand;
};

struct TrafficState {
    const TrafficDemand *demands;
    int32_t numDemands;
};
```

如果拓扑和流量都需要被每个 world 共享，也可以把它们都放进 `WorldInit` 中。

### 5.2 在 bindings.cpp 中接收 Python 传入的数据

类似当前 `walls / rewards / end_cells` 的方式，在 Python binding 构造函数中增加：
- topology arrays
- traffic arrays

建议输入仍优先使用：
- `numpy.ndarray`
- C contiguous
- 明确 dtype

例如：
- `int32` 的边端点数组
- `float32` 的容量 / 流量数组

在 `bindings.cpp` 中先把这些 NumPy 输入整理为 host 侧连续 buffer。

### 5.3 在 mgr.cpp 的 CUDA 分支中手动申请 GPU 空间并上传

这一步直接套用 `GridState` 模板：

1. 计算所需字节数
2. `cu::allocGPU(...)`
3. 在 host 侧构造 staging struct，其中内部指针指向 GPU 子区域
4. `cudaMemcpy` 复制 struct 本体和底层数组

伪代码：

```cpp
char *topology_data = (char *)cu::allocGPU(sizeof(TopologyState) +
                                           sizeof(LinkData) * num_links);
LinkData *gpu_links = (LinkData *)(topology_data + sizeof(TopologyState));

TopologyState topo_staging {
    .links = gpu_links,
    .numLinks = num_links,
    .numNodes = num_nodes,
};

cudaMemcpy(topology_data, &topo_staging, sizeof(TopologyState),
           cudaMemcpyHostToDevice);
cudaMemcpy(gpu_links, cpu_links, sizeof(LinkData) * num_links,
           cudaMemcpyHostToDevice);
```

流量数据同理。

### 5.4 通过 WorldInit 把 GPU 指针传入 world

参考 `src/init.hpp:12-15`：

```cpp
struct WorldInit {
    EpisodeManager *episodeMgr;
    const GridState *grid;
};
```

可以扩展成：

```cpp
struct WorldInit {
    EpisodeManager *episodeMgr;
    const GridState *grid;
    const TopologyState *topology;
    const TrafficState *traffic;
};
```

然后在 `setupWorldInitData(...)` 中把这些指针填进去。

### 5.5 在 Sim / tick / 其他 system 中读取

参考 `src/sim.cpp:38`：

```cpp
const GridState *grid = ctx.data().grid;
```

未来可以变成：

```cpp
const TopologyState *topology = ctx.data().topology;
const TrafficState *traffic = ctx.data().traffic;
```

这样 GPU 上的系统代码就能直接访问外部传入的数据。

---

## 6. 如果某个字段需要“Python 每步写、GPU 每步读”，就不要走 GridState 模式，而要走 exportColumn 模式

这是最容易混淆的点。

### 适合 exportColumn 的情况

如果你要实现的是：
- Python 每一步都更新某些输入
- step 前写入，step 中 GPU 立即读取
- 并且这个数据天然属于某个 entity / world 的状态列

那么应当按 `Action` 的方式实现：

1. 在 `types.hpp` 新增 component
2. 加入对应 archetype
3. 在 `Sim::registerTypes()` 中 `exportColumn<...>`
4. 在 `ExportID` 中新增 slot
5. 在 `Manager` 中新增 `xxxTensor()`
6. 在 Python wrapper 中 `.to_torch()` 保存句柄
7. Python 直接写 tensor
8. `tick()` 直接读 component 引用

### 例子：为什么 Action 适合 exportColumn

因为 action 是：
- 每个 world / agent 一份
- 每步更新
- step 时直接消费

### 例子：为什么 topology 通常不适合 exportColumn

因为 topology 常常是：
- 全局共享
- 结构复杂
- 不属于某个单独 agent 的一列属性
- 初始化后大部分时间不变

---

## 7. 其他 Agents 在实现时必须保持的对齐关系

如果走 **exportColumn 模式**，必须保持下面 4 处同步：

1. `src/types.hpp`
   - 新组件定义
   - `ExportID` 枚举顺序
2. `src/sim.cpp`
   - `registerComponent`
   - `registerArchetype`
   - `exportColumn`
3. `src/mgr.cpp`
   - `Manager::xxxTensor()` 的 slot、dtype、shape
   - `numExportedBuffers = (uint32_t)ExportID::NumExports`
4. `src/madrona_simple_example/gridworld.py`
   - `.to_torch()` 句柄导出
   - Python 调用方读写方式

如果走 **共享 GPU 结构体模式**，必须保持下面 4 处同步：

1. 原生结构体定义（类似 `grid.hpp`）
2. `bindings.cpp` 中的 Python 输入解析和 host staging buffer 组装
3. `mgr.cpp` 中 CPU / GPU 两套初始化逻辑
4. `init.hpp` / `sim.hpp` / `sim.cpp` 中 world init 传递与系统读取路径

---

## 8. 这件事和“通过 PyTorch 作为中间桥梁”之间的准确关系

你原始描述里提到：

> 通过 PyTorch 作为中间桥梁，将 Python 字段中的数据通过 PyTorch 传输到 GPU 的数据块中。通过 exportcolumn 和对应的类型。

这里需要区分两种情况：

### 情况 A：exportColumn 对应的 ECS 列

这时你的描述是准确的。

路径就是：

```text
Python 字段
→ torch tensor
→ exported ECS column
→ GPU 上的组件列
```

`Action` / `Reset` 就是这样工作的。

### 情况 B：manager 手动申请的共享 GPU 结构体

这时 **并不是必须经过 PyTorch**。

当前 `GridState` 的真实路径是：

```text
Python numpy
→ nanobind / C++
→ host staging struct
→ cudaMemcpy
→ GPU 结构体 / 数组
```

这里没有 `.to_torch()` 参与。

所以给其他 Agents 的指导应该写清楚：

- **动态、逐步写入的 ECS 输入**：优先走 PyTorch + exportColumn
- **初始化时上传的大块共享仿真数据**：优先走 bindings + Manager + cudaMemcpy

这两者都属于“从 Python 接口把数据送到 GPU”，但技术路径不同。

---

## 9. 对你当前目标最合适的落地方案

如果目标是让 Madrona 支持读取：
- 拓扑数据文件
- 流量数据文件

推荐拆成两层：

### 第一层：Python 负责文件读取和预处理

由 Python 完成：
- 文件解析
- 清洗
- 转成标准 numpy arrays / torch tensors

这样 C++ / CUDA 侧就不需要关心文件格式。

### 第二层：Madrona 只接收结构化数组

通过 Python binding 把整理好的数组传给 native 层：
- 对静态拓扑 / 流量基表：走 `GridState` 类似路径
- 对需要每步更新的控制量：走 `Action` 类似路径

这种职责划分最清晰，也最符合当前仓库已有结构。

---

## 10. 给其他 Agents 的一句话执行准则

如果要把外部 Python 数据送进 Madrona，请先判断该数据属于哪一类：

- **每步由 Python 改写，并由系统按组件列消费** → 参照 `Action`，使用 `exportColumn + to_torch()`
- **初始化时一次性上传、由多个 world / system 共享读取** → 参照 `GridState`，使用 `bindings.cpp + mgr.cpp 中手动 GPU 分配与 cudaMemcpy`

对于“拓扑数据 / 流量数据文件接入 Madrona”这个目标，默认优先采用第二种路径；只有在某些流量字段需要逐 step 更新时，再把那部分单独拆成第一种路径。
