# AGENTS · 不变项（跨轮共享，几乎不变）

- 项目：madrona_simple_example，madrona流仿真 + LLM Serving/训练负载仿真；分支 `llm-inferebce-v3-multiworld`（用户原拼写）从 `llm-inference-v2` 开出。
- 基线：32rank 1P+1D Serving（P 0-15/D 16-31、n32_4leaf、request 4×4096×8、op_profile 233行stage=1），单world CPU/GPU RUN OK（9115步/33秒sim）。
- 红线：容量常量可调、仿真逻辑不动（002/003惯例）；不引入未确认新依赖；不破坏pytest（43+25）与现有RUN OK。
- 约定：每轮目录 `docs/runs/NNN-<功能名>/`，根 `PRD.md`/`CHECKPOINT-REPORT.md` 是指针；决策进 `DECISIONS.md` 带轮次标记；跨轮知识进 `docs/decisions-archive/`，研究结论进 `docs/research/` 并回填PRD清单。
- 报告口径：照11号汇报版结构（配置/输入/输出对照/全量估算/不确定度）；凡落到轮次目录之外的正式文档（如 /app/report/ 下的汇报版），必须同时在 `docs/runs/NNN-<功能名>/` 存一份同名终审版。
