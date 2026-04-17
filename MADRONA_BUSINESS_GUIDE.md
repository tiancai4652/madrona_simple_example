# Madrona 业务编码说明

本文基于本仓库的最小 GridWorld 示例，总结一套“只看业务代码”的开发方式。目标是：拿到一个新业务后，只需要围绕组件、实体、系统、任务图四件事扩展代码，就能快速把业务逻辑接到 Madrona 中。

本文**不展开** DoD 细节、底层 ECS 实现细节、CPU/GPU 执行器细节；只关注业务代码应当怎么写。

## 1. 业务开发主要看哪些文件

- `src/types.hpp`：定义组件类型、业务枚举、实体 archetype
- `src/sim.hpp`：定义 world 类型 `Sim`、业务配置 `Sim::Config`、上下文 `Engine`
- `src/sim.cpp`：注册组件、创建实体、实现系统、组装任务图
- `src/grid.hpp` / `src/init.hpp`：放 world 共享的业务初始化数据
- `src/bindings.cpp` / `src/mgr.cpp` / `src/madrona_simple_example/gridworld.py`：宿主接口层；只有当业务要和 Python/Torch 交换输入输出时才需要改

如果你只是在做业务逻辑，平时主要改 `src/types.hpp` 和 `src/sim.cpp`。

## 2. 先建立一个统一心智模型

在这个示例里，业务代码可以分成 4 层：

1. **组件（Component）**：实体身上保存的状态
2. **实体定义（Archetype）**：一个业务对象由哪些组件组成
3. **系统（System）**：每一步如何读写这些组件
4. **任务图（Task Graph）**：这些系统以什么顺序执行

可以把它理解成：

- `types.hpp` 回答“业务对象有什么状态”
- `Sim` 构造函数回答“世界开始时创建哪些对象”
- 系统函数回答“每一步怎么推进状态”
- `setupTasks` 回答“先跑哪些逻辑，再跑哪些逻辑”

## 3. 组件类型怎么定义

### 3.1 组件的职责

组件只负责**存状态**，不要在组件里塞业务逻辑。

当前示例中的组件都在 `src/types.hpp`：

- `Reset`：是否需要强制重置
- `Action`：当前动作输入
- `GridPos`：当前位置
- `Reward`：奖励输出
- `Done`：回合是否结束
- `CurStep`：当前回合步数

这些组件有一个共同特点：

- 都是简单结构体或枚举
- 字段很少
- 一眼能看出业务含义
- 不包含方法

### 3.2 推荐写法

推荐把组件写成非常直接的 POD：

```cpp
struct HP {
    int32_t v;
};

struct Position {
    int32_t y;
    int32_t x;
};

enum class MoveIntent : int32_t {
    Up,
    Down,
    Left,
    Right,
    None,
};
```

### 3.3 组件设计规范

- **一个组件表达一类状态**，不要把很多无关字段堆到一个大结构里
- **名称用业务名词**，例如 `HP`、`Position`、`AttackCooldown`
- **离散输入/状态优先用 `enum class`**
- **共享只读数据不要放组件里**，放到 `WorldInit` 或 `Sim` 成员里；本示例中的网格地图 `grid` 和 `maxEpisodeLength` 就属于 world 级共享数据
- **需要被外部读取/写入的字段，才考虑后续导出**

### 3.4 GPU 业务代码中的数据结构约束

由于业务系统要在 GPU 上运行，业务代码应当按“受限 C++ 子集”来写。

这意味着：

- **不要依赖不定长容器**，例如 `std::list`、`std::vector`、`std::map`
- **不要依赖 STL 风格接口和高级标准库能力**，例如 `begin` / `end`、复杂迭代器算法、动态分配驱动的数据结构
- **优先使用定长数组、POD 结构体、`for` / `while` / `if` / `switch` 这类简单语法**
- **业务数据结构要在编译期就尽量确定上限**

对业务开发来说，最重要的一条是：

> 凡是你在普通 C++ 里想用“动态容器”表达的业务状态，在 Madrona 里通常都要改写成“定长容量 + 当前长度”的结构。

### 3.5 如何在 Madrona 里表达 list 这类不定长结构

如果业务上有类似下面的需求：

```cpp
std::list<int> list;
```

在 Madrona 业务代码里，通常不能直接这样建模，而是要先估计它的**最大容量**，然后改成定长数组。

例如：

```cpp
#define MAX_LIST_LENGTH 1000

struct MyList {
    uint32_t numElems;
    uint32_t data[MAX_LIST_LENGTH];
};
```

这里的设计含义是：

- `MAX_LIST_LENGTH`：这个业务字段允许的最大元素个数
- `numElems`：当前实际使用了多少个元素
- `data`：真正存储内容的定长数组

也就是说，原来“不定长 list”的语义，被改写成：

> 一个最大长度固定、当前长度可变的数组容器。

### 3.6 这种定长数组结构怎么用

常见写法如下：

```cpp
inline void clearList(MyList &list)
{
    list.numElems = 0;
}

inline bool pushBack(MyList &list, uint32_t v)
{
    if (list.numElems >= MAX_LIST_LENGTH) {
        return false;
    }

    list.data[list.numElems] = v;
    list.numElems += 1;
    return true;
}

inline void removeAt(MyList &list, uint32_t idx)
{
    if (idx >= list.numElems) {
        return;
    }

    for (uint32_t i = idx + 1; i < list.numElems; i++) {
        list.data[i - 1] = list.data[i];
    }

    list.numElems -= 1;
}
```

也就是：

- 新增元素：手动写入 `data[numElems]`
- 删除元素：自己用 `for` 循环搬移后续元素
- 遍历元素：只遍历 `[0, numElems)`

例如遍历：

```cpp
for (uint32_t i = 0; i < list.numElems; i++) {
    uint32_t v = list.data[i];
    // 处理 v
}
```

### 3.7 针对不定长数据的建模规范

以后只要业务里出现“数量不固定”的东西，例如：

- 背包里的物品列表
- 技能目标列表
- 邻居列表
- 命中结果列表
- 某实体当前帧的事件集合

统一按下面思路建模：

1. 先估计一个合理最大值 `MAX_*`
2. 用定长数组保存数据
3. 单独维护一个当前长度字段
4. 所有插入、删除、遍历都手写数组逻辑
5. 超上限时，明确业务策略：拒绝写入、截断、覆盖，三选一

推荐模式：

```cpp
#define MAX_TARGETS 64

struct TargetList {
    uint32_t count;
    Entity targets[MAX_TARGETS];
};
```

不要只定义数组，不记录当前长度；否则系统无法区分“有效数据”和“未使用槽位”。

### 3.8 业务代码语法约束

在业务系统代码中，建议遵守下面这组硬约束：

- **只使用简单结构体、枚举、数组**
- **只使用 `for`、`while`、`if`、`switch` 这类基础控制流**
- **避免使用 C++ 标准库容器和复杂算法接口**
- **避免写依赖动态内存语义的业务逻辑**
- **业务核心逻辑尽量写成平铺、直接、可展开的代码**

可以理解成：

> Madrona 的业务代码更接近“可在 GPU 上稳定执行的显式数组编程”，而不是“桌面 C++ 中自由使用 STL 的面向对象编程”。

## 4. 实体怎么定义

### 4.1 实体定义的本质

在这个示例中，实体不是通过写一个复杂类来定义，而是通过 **archetype = 组件集合** 来定义。

`src/types.hpp` 中的示例：

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

这段代码表达的是：

> 一个 `Agent` 实体，天然带有 `Reset / Action / GridPos / Reward / Done / CurStep` 这组业务状态。

### 4.2 实体定义规范

定义一个实体时，先回答两个问题：

1. 这个业务对象在每一步必须保存哪些状态？
2. 哪些系统会处理它？

然后把这些状态组织成一个 archetype。

推荐做法：

- **一个 archetype 对应一个明确业务角色**，例如 `Agent`、`Bullet`、`Monster`
- **不要为了“以后可能会用”提前塞很多组件**
- **如果两个业务对象生命周期和状态明显不同，就拆成两个实体 archetype**

## 5. 如何创建一个实体

### 5.1 当前示例的创建方式

实体是在 `Sim` 构造函数里创建的：

```cpp
Entity agent = ctx.makeEntity<Agent>();
ctx.get<Action>(agent) = Action::None;
ctx.get<GridPos>(agent) = GridPos {
    grid->startY,
    grid->startX,
};
ctx.get<Reward>(agent).r = 0.f;
ctx.get<Done>(agent).episodeDone = 0.f;
ctx.get<CurStep>(agent).step = 0;
```

这说明业务层的创建流程非常简单：

1. 用 `ctx.makeEntity<实体类型>()` 创建实体
2. 用 `ctx.get<组件>(entity)` 给每个组件写初始值

### 5.2 创建实体时的规范

- **实体创建集中放在 `Sim` 构造函数**，让 world 初始化逻辑一处可见
- **创建完后立即写完整初始状态**，不要依赖“后面某个系统顺手补齐”
- **初始值来自 world 共享配置时，直接从 `init` 或 `ctx.data()` 读取**
- **一个 world 内需要几个业务对象，就在构造函数里明确创建几个**

如果以后你的业务不止一个 `Agent`，也建议仍然按这个模式创建，只是增加不同 archetype 或多个实体实例。

## 6. 系统怎么定义

### 6.1 当前示例的系统写法

当前业务核心系统是 `src/sim.cpp` 里的 `tick`：

```cpp
inline void tick(Engine &ctx,
                 Action &action,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step)
```

这类系统函数有两个核心特征：

- 第一个参数是 `Engine &ctx`，用来访问 world 级共享数据
- 后面的参数是这个系统要读写的组件引用

在这个示例里，`tick` 完成了完整的一步业务推进：

- 读取动作
- 计算新位置
- 处理边界和墙
- 判断终点/重置/步数上限
- 回写 `GridPos / Reward / Done / CurStep`

### 6.2 系统设计规范

推荐把系统写成“**一个函数只做一段明确业务阶段**”：

- 输入处理
- 移动/战斗/结算
- 奖励计算
- 结束判定
- 重置

当前示例为了最小化，只用了一个 `tick` 把这些事情串在一起；业务复杂后，建议拆成多个系统函数，但每个函数仍保持这种签名风格。

### 6.3 系统编码规范

- **系统名用动词**，例如 `tick`、`moveAgent`、`applyDamage`、`resetEpisode`
- **参数列表就是系统的数据依赖声明**：需要什么组件，就把什么组件写进参数里
- **world 共享只读数据通过 `ctx.data()` 访问**，不要复制到每个实体组件里
- **系统内部直接做业务状态变更**，不要把逻辑分散到组件方法里
- **一个系统尽量只负责一个业务阶段**，方便排查和重排任务图

## 7. 系统是如何“使用实体”的

这是最重要的一点：

> 在这个编码模式里，系统通常不是“按实体类名写逻辑”，而是“按组件集合写逻辑”。

以当前示例为例：

- `Agent` 实体包含 `Action / Reset / GridPos / Reward / Done / CurStep`
- `tick` 系统正好声明了这些组件参数
- 所以这个系统就会作用到 `Agent` 这样的实体上

也就是说，**实体是否会被某个系统处理，取决于它是否具备该系统要求的组件集合**。

这对业务开发有两个直接结论：

1. 如果你想让某类实体进入某个系统，就让它带上该系统需要的组件
2. 如果你不想让某类实体进入某个系统，就不要让它拥有那组组件，或者拆成另一个系统

因此，业务建模时最重要的不是“写多少继承层级”，而是：

- 这个业务对象有哪些状态
- 哪些系统应该处理它
- 这些系统依赖哪些组件

## 8. 任务图是如何组成的

### 8.1 当前示例的任务图

任务图组装在 `Sim::setupTasks`：

```cpp
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);
    builder.addToGraph<ParallelForNode<Engine, tick,
        Action, Reset, GridPos, Reward, Done, CurStep>>({});
}
```

这段代码表达的是：

1. 初始化一个 step 用的任务图
2. 往图里加入一个并行节点
3. 这个节点执行 `tick` 系统
4. 节点处理的对象，是那些具备 `Action / Reset / GridPos / Reward / Done / CurStep` 的实体

### 8.2 如何理解业务任务图

对业务开发来说，可以把任务图理解成：

> “一帧/一步内，系统按什么顺序执行”

本示例的任务图很简单：

- 只有一个系统 `tick`
- 所以一步就是执行一次 `tick`

如果业务复杂，可以按下面的顺序拆分：

1. 读取输入
2. 更新移动/行为
3. 处理碰撞或交互
4. 计算奖励
5. 判断结束
6. 执行重置

然后把这些系统按顺序加入 `setupTasks`。

### 8.3 任务图编排规范

- **任务图顺序要和业务阶段一致**，让人一眼能看懂一步发生了什么
- **优先拆成多个短系统，而不是一个巨大的总系统**
- **每个节点只关心当前阶段需要的组件**
- **先决定业务阶段，再决定节点拆分**；不要为了底层机制反过来扭曲业务表达

## 9. 业务代码与宿主接口怎么分工

本仓库里还有一层宿主接口代码：

- `src/bindings.cpp`：把外部输入转换成 `Manager` 初始化参数
- `src/mgr.cpp`：驱动 CPU/GPU 执行器，并把导出列包装成 Tensor
- `src/madrona_simple_example/gridworld.py`：提供 Python 侧易用接口

对业务开发来说，可以按下面理解：

- **业务状态本身**：在组件里定义
- **业务每步逻辑**：在系统里定义
- **外部怎么喂数据、怎么拿结果**：在 bindings/manager/python 层定义

只有当你要把业务状态暴露给外部时，才需要增加导出。

当前示例在 `registerTypes` 里通过 `exportColumn` 导出了：

- `Reset`
- `Action`
- `GridPos`
- `Reward`
- `Done`

因此 Python 侧可以直接读写这些张量。

## 10. 推荐的业务开发步骤

以后接一个新业务时，建议按这个顺序改：

### 第一步：定义组件

先把业务状态拆出来，写进 `src/types.hpp`。

例如：

- 位置
- 朝向
- 血量
- 技能冷却
- 攻击意图
- 回合状态

### 第二步：定义实体 archetype

把一个业务对象需要的组件组合成 archetype。

例如：

```cpp
struct Hero : public madrona::Archetype<
    Position,
    HP,
    MoveIntent,
    AttackCooldown
> {};
```

### 第三步：注册类型

在 `Sim::registerTypes` 中注册新增组件和 archetype：

```cpp
registry.registerComponent<Position>();
registry.registerComponent<HP>();
registry.registerComponent<MoveIntent>();
registry.registerComponent<AttackCooldown>();

registry.registerArchetype<Hero>();
```

如果要给外部读写，再按需 `exportColumn`。

### 第四步：创建实体并初始化

在 `Sim` 构造函数里创建实体并写初始值：

```cpp
Entity hero = ctx.makeEntity<Hero>();
ctx.get<Position>(hero) = Position {0, 0};
ctx.get<HP>(hero).v = 100;
ctx.get<MoveIntent>(hero) = MoveIntent::None;
ctx.get<AttackCooldown>(hero).frames = 0;
```

### 第五步：实现系统

把业务流程拆成系统函数：

```cpp
inline void moveHero(Engine &ctx,
                     MoveIntent &intent,
                     Position &pos)
{
    // 业务逻辑
}
```

如果还有战斗、奖励、重置，就继续拆新的系统。

### 第六步：把系统接进任务图

在 `setupTasks` 里按业务顺序组装：

```cpp
TaskGraphBuilder &builder = taskgraph_mgr.init(0);

builder.addToGraph<ParallelForNode<Engine, moveHero,
    MoveIntent, Position>>({});
```

后续系统也按顺序继续加入。

## 11. 一套适合业务开发的简明规范

最后把上面的做法压缩成一套可以直接执行的规范：

### 11.1 组件规范

- 组件只存状态，不写行为
- 组件名用业务名词
- 用小而清晰的结构体表达状态
- 离散状态优先用 `enum class`

### 11.2 实体规范

- 实体 = 一组组件的组合
- 一个实体 archetype 代表一个明确业务角色
- 不提前塞无关组件

### 11.3 系统规范

- 系统 = 一个明确业务阶段
- 系统名用动词
- 参数列表显式声明依赖的组件
- world 共享数据通过 `ctx.data()` 获取

### 11.4 任务图规范

- `setupTasks` 按业务阶段顺序组装节点
- 简单业务可先用一个总系统
- 复杂业务优先拆成多个短系统
- 顺序设计要体现“这一步业务如何流动”

### 11.5 初始化规范

- 所有实体在 `Sim` 构造函数中创建
- 创建后立刻写完整初始状态
- 共享配置和静态地图数据放 world 级，而不是复制到每个实体

## 12. 用一句话总结这套模式

在这个示例中，Madrona 的业务开发可以总结成一句话：

> 先定义业务状态（组件），再把状态组合成业务对象（实体），再用系统函数推进状态，最后在任务图里排好这些系统的执行顺序。

如果你按这个顺序写，一个新业务通常只需要关注：

- `types.hpp` 里有什么状态
- `Sim` 构造函数里创建了什么对象
- `sim.cpp` 里有哪些系统
- `setupTasks` 里这些系统按什么顺序跑

这样就足够开始做业务开发了。
