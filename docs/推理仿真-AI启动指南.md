# 推理仿真 AI 启动指南（给 AI 看的 onboarding 文档）

> 用法：新同事把本文件全文丢给任意 AI（连同本仓库路径），说一句"按此文档帮我把推理仿真跑起来"即可。
> AI 按 §4→§5→§6 顺序执行，跑通即停，不要自由发挥改仿真逻辑。

---

## 1. 你是谁、要做什么

你是接手本仓库推理仿真研究的 AI 助手。你的唯一目标：**协助用户把推理（Serving / PD 分离）仿真在 CPU 上跑通（RUN OK），并能解释输入/输出**。

- 仓库地址：`https://github.com/tiancai4652/madrona_simple_example.git`，当前分支 `llm-inferebce-v3-multiworld`（注意是用户原拼写，不要"纠正"成分支名）。新同事先执行：
  ```bash
  git clone --recursive https://github.com/tiancai4652/madrona_simple_example.git
  cd madrona_simple_example
  git checkout llm-inferebce-v3-multiworld
  ```
  （必须 `--recursive`，`external/madrona` 是子模块；本地已有 `/app/madrona_simple_example` 时直接在该路径工作，不必重 clone。）
- 项目一句话：基于 Madrona ECS 的流级网络仿真器 + Chakra 系统层 + Inference（Prefill/Decode/KV Transfer）运行时，做 LLM 训练/推理负载仿真。
- 基线（已验证 RUN OK，不要质疑，直接复跑）：
  - 32 rank、1 个 Prefill worker（rank 0-15）+ 1 个 Decode worker（rank 16-31）
  - topo `n32_4leaf`，request 4×4096×8，op_profile 233 行 stage=1
  - CPU 全量：**finished step=9115，sim_time≈33038.9ms（≈33秒仿真时间），4/4 请求 Finished，零 FAILED，两次连跑逐位一致**

## 2. 红线（AI 必须遵守）

1. **仿真逻辑不动**：只允许调容量常量（如 `NPU_NUM`、`MAX_BUFFER_CHUNKS`），不改 `src/sim_systems_*.cpp`、`src/sys/` 下的推进/调度/KV 算法。
2. **不引入新依赖**：只用仓库已有 Python/C++ 依赖（numpy/torch/cmake/CUDA 已就绪）。
3. **不破坏现有绿线**：任何改动后必须 `pytest` 全绿（43 passed + 25 subtests），且基线命令仍 RUN OK。
4. **只读先行**：改任何 `src/` 文件前，先读文件、先报备用户，经批准再改。轮次目录 `docs/runs/` 下的归档只读，不要覆盖。

## 3. 先读这 5 个文件（按顺序，读完再动手）

1. `README.md` §1-§7 —— 架构（网络层+系统层+耦合点 `setFlow/checkFlowFinish`）与运行入口。
2. `AGENTS.md` —— 不变项（基线口径/红线/轮次约定），全文仅 7 行。
3. `scripts/run_serving_simulator.py --help` —— 推理仿真入口参数（HEAD 版 `--workload-params` bug 已修，直接用它；不要用 `docs/runs/002-*/run_serving_ds671b.py` 那个轮次内 workaround 副本）。
4. `docs/runs/002-DeepSeekV3-671B训练负载/request.json` —— 基线命令与输入清单（`command` 键即全量命令）。
5. `docs/runs/002-DeepSeekV3-671B训练负载/过程记录.md` §6.2-§6.4、§七、§九 —— request_trace 取值理由、serving_config 语义（MLA/latent 576/ref 4096）、D 侧副本构造、MAX_BUFFER_CHUNKS 扩容到 128 的原因。

代码地图（按需查，不用全读）：

| 位置 | 内容 |
| --- | --- |
| `scripts/run_serving_simulator.py` | 推理仿真唯一入口（topo+workload+request_trace+op_profile → step 循环 → `--out-csv`） |
| `src/madrona_simple_example/gridworld.py` | `GridWorld` 包装（喂数/step/读状态/写 CSV） |
| `src/madrona_simple_example/inference/` | request_trace 与 op_profile 的 loader（字段校验真相以此为准） |
| `src/sys/` | 系统层（Chakra 调度+KV transfer），只读 |
| `src/sim_systems_*.cpp`、`src/sim_tasks.cpp` | 网络层 12 阶段流水线，只读 |
| `tests/test_inference_config.py`、`tests/test_inference_integration.py` | 回退门（43+25） |

## 4. 环境检查与全新构建（AI 从这里开始）

### 4.0 全新 clone 全流程（`build/` 不存在时执行；已有 `build/` 则跳到 §4.1）

```bash
# 0. 依赖确认（任一缺失先装好再往下）
cmake --version   # 期望 >= 3.18
python3 --version # 期望 3.10（本基线验证环境）
g++ --version     # C++ 编译器即可

# 1. 拉代码（必须 --recursive，external/madrona 是子模块）
git clone --recursive https://github.com/tiancai4652/madrona_simple_example.git
cd madrona_simple_example
git checkout llm-inferebce-v3-multiworld

# 2. 构建（CPU 后端；容量常量 NPU=32/BUFFER=128 已在 HEAD 里，无需改源码）
mkdir -p build && cmake -S . -B build && cmake --build build -j$(nproc)
# 期望：BUILD 成功，0 error（唯一既有 warning 是 sys_types.hpp:788 setFinish 未用参数，无视）

# 3. 安装 Python 绑定
pip install -e .
python3 -c "import madrona_simple_example; print('import ok')"
# 期望：import ok
```

### 4.1 存量环境检查（AI 先跑这 3 条，任一失败先修环境再往下）

在仓库根目录下执行（存量环境是 `/app/madrona_simple_example`，新 clone 则在 clone 出来的目录）：

```bash
git branch --show-current
# 期望：llm-inferebce-v3-multiworld

ls docs/runs/002-DeepSeekV3-671B训练负载/topo_leafspine_h32.txt \
   docs/runs/002-DeepSeekV3-671B训练负载/workload_pd32 \
   docs/runs/002-DeepSeekV3-671B训练负载/request_trace.json \
   docs/runs/002-DeepSeekV3-671B训练负载/op_profile.json
# 期望：4 个路径都存在（workload_pd32 下应有 npu.0..31.json 共 32 个）
# 注：topo 是 `/app/exps/inputs/topo/n32_4leaf/leafspine_h32_topo.txt` 的逐字节副本（md5 ab94efa4…），已入库，开箱即用。

python3 -m pytest tests/test_inference_config.py tests/test_inference_integration.py -q
# 期望：43 passed, 25 subtests passed
```

构建产物已就绪（`build/` 存在）时**不要重编**。只有改了 `src/` 容量常量才重编：`cmake --build build -j32`，重编后重跑上一条 pytest。

## 5. 先跑冒烟（~30 秒，链路验证）

```bash
python3 docs/runs/002-DeepSeekV3-671B训练负载/probe_pd32.py 5000 \
  docs/runs/002-DeepSeekV3-671B训练负载/request_trace_smoke.json
```

- 期望：`finished=True`，请求 `state=7=Finished`，`p_finish_ns>0` 且 `kv_start_ns>0`，`finished_npus=32/32`，无 FAILED。
- 若冒烟失败：把完整报错贴给用户，不要继续跑全量。

## 6. 再跑全量基线（~1 分钟，RUN OK 验证）

 exact 基线命令（`request.json` 的 `command` 键，CPU 模式）：

```bash
python3 scripts/run_serving_simulator.py \
  --topo docs/runs/002-DeepSeekV3-671B训练负载/topo_leafspine_h32.txt \
  --workload docs/runs/002-DeepSeekV3-671B训练负载/workload_pd32 \
  --request-trace docs/runs/002-DeepSeekV3-671B训练负载/request_trace.json \
  --workload-params docs/runs/002-DeepSeekV3-671B训练负载/op_profile.json \
  --npu-count 32 --ring-dims 32,1,1 --chunks 1 \
  --prefill-workers 0-15 --decode-workers 16-31 \
  --p-max-batch 16 --p-max-tokens 8192 --d-max-batch 16 --d-max-tokens 32768 \
  --num-layers 61 --num-kv-heads 128 --head-dim 128 --bytes-per-elem 2 \
  --kv-partition-factor 0 \
  --prefill-reference-tokens 4096 --decode-reference-tokens 4096 \
  --kv-mode mla --kv-latent-dim 576 \
  --max-steps 200000 --print-every 500 \
  --out-csv /tmp/inference_stats_repro.csv
```

- 输入语义速查：request=4 条（arrival 0/1/2/3ms，prompt 4096，output 8）；P worker 0-15 / D worker 16-31（各跑完整 16-rank workload，D 侧是 +16 偏移副本，零跨边——这是 002 试错后定案的 1P+1D 闭合方案）；MLA latent 576（=512+64），单请求 KV=61×4096×576×2=287834112B；decode ref 取 4096 使线性回退中性（模板本就是 seq4096 实测值）。
- 期望输出：`finished at step 9115, sim_time_ms=33038.9020`，CSV 中 4/4 请求 `state=7=Finished`、`output_done=8/8`、`kv_shards=16/16`、`kv_bytes=287834112`，无 FAILED。墙钟约 50 秒。
- 若输出与期望逐位一致 → **RUN OK，任务完成**，向用户汇报三行：step/sim_time/CSV 结论 + pytest 绿 + 未改任何源码。

## 7. 常见坑（AI 先自查，不要问用户）

| 现象 | 原因/处理 |
| --- | --- |
| `AttributeError: workload_params_table` | 用了旧版入口脚本。改用 HEAD 的 `scripts/run_serving_simulator.py`（已修），不要用 002 轮次目录里的 `run_serving_ds671b.py` 副本。 |
| `FATAL ... MAX_BUFFER_CHUNKS=16 ... capacity exceeded` | 旧二进制/旧常量。当前 HEAD 已是 128（`src/types.hpp:141`），重编即可；不要自己改小负载绕过去。 |
| 跑 20 万步未 finish、无 FAILED、time 以 1us/step 爬 | 误用了 P 0-7/D 8-15 切半划分（训练 DAG 跨边断裂，已证伪）。必须用 0-15/16-31 的 1P+1D 闭合命令（§6 原样抄）。 |
| GPU 相关报错 | 本指南只要求 CPU。GPU 需另约卡、另开流程，不要在 CPU 任务里顺手试 GPU。 |
| 想调请求数/prompt/output | 允许，但先保证 §6 基线 RUN OK。改 trace 只改 `request_trace.json` 的副本（放 /tmp），不碰归档原文件。 |

## 8. 向用户汇报的格式（跑完照此三段说）

1. **结论**：RUN OK / 未 RUN OK（step、sim_time、Finished 数、FAILED 数）。
2. **证据**：实际命令、CSV 路径、pytest 结果、有无改动源码（`git status --short` 贴出来）。
3. **下一步建议**：若 RUN OK，问用户要研究哪个方向（改负载规模 / 换 topo / 看 KV 流量 / 跑多 world）；若失败，贴完整报错并停下等指示。

---

*基线依据：`docs/runs/002-DeepSeekV3-671B训练负载/过程记录.md` §九（9115 步/33 秒 sim/两次一致）、`request.json`（命令与输入清单）、`DECISIONS.md` [002] 行（NPU 16→32、MAX_BUFFER_CHUNKS 16→128 两次容量批准）。本文档只管"跑起来"，不等同于 D 侧语义复核（002 遗留开放项，研究阶段再议）。*
