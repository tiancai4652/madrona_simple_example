#!/bin/bash

# GPU 版运行脚本：结构参照 multiverse-dev/run.sh，命令参照 run_flow_simulator.sh。
# 用法：bash run_flow_simulator_gpu.sh > 1.log 2>&1

set -euo pipefail

ROOT=/app/madrona_simple_example
CUDA_HOME=/usr/local/cuda-12.8

# ============================================================
# 配置参数：直接在这里修改参数值
# ============================================================
DIM0=2
DIM1=2
DIM2=4
NPU_NUM=16
DEV_MODE=2   # 0 = SYS, 1 = NET, 2 = MIX
GPU_ID=0     # 物理 GPU 编号（自动 export 为 CUDA_VISIBLE_DEVICES，映射为逻辑 GPU0）

# 日志开关（对应 multiverse global_types.hpp；SIMPLE_LOG_MODE 与 SYS_LOG 二选一）
SIMPLE_LOG_MODE=0   # 1 = 简单日志（需 SYS_LOG=0）
SYS_LOG=0           # 1 = 详细日志（需 SIMPLE_LOG_MODE=0）
SYS_LOG_SPECIAL=0
SYS_LOG_TARGET_NODE=0   # 只打印该 NPU 的日志

# 流仿真器网络日志（std::cout 的 [SYS][...] 输出；开启会自动启用 trace 模式）
NET_LOG=0               # 1 = 开启流仿真器网络日志
NET_LOG_SCOPE=all
NET_LOG_EVERY=10

run_log=
topology_file=/app/jiuding_dodsim/examples/leafspine128/leafspine_h128_topo.txt
workload_path="/app/华为负载/20260715 - 九鼎仿真平台训练负载示例/test_16a2_8p_deepseekv3_671b_l2_train_gbs16_seq4096_tp2_ep4_pp2/workload_train1.1_chakra"
max_steps=${MAX_STEPS:-200000}
PRINT_EVERY=${PRINT_EVERY:-10}
out_csv=${OUT_CSV:-}
# GPU 编译模式：0 = 正式(LTO)，1 = Debug 快速编译（首次打通/调试用，浮点不保证一致）
GPU_COMPILE_BOOTSTRAP=0
# initTasks 诊断：0=正常，1=空内核，2=仅 setupTasks，3=仅 constructGraphs
GPU_INIT_TASKS_DIAG_STAGE=0
GPU_TRACE_INIT=${GPU_TRACE_INIT:-0} # 1 = 打印并同步每个 GPU 初始化阶段
# Madrona 默认 4GB device malloc heap 会使首个 malloc 内核 launch 资源不足。
# TaskGraph 构建仅需少量临时分配，64MB 足够。
GPU_DEVICE_HEAP_SIZE=67108864
# CHAKRA_NODES_DATA_LENGTH / MAX_CHAKRA_NODES_PER_NPU 留空则按 workload 自动计算
CHAKRA_NODES_DATA_LENGTH=
MAX_CHAKRA_NODES_PER_NPU=
# ============================================================

echo "=== 配置参数 ==="
echo "dim0: $DIM0"
echo "dim1: $DIM1"
echo "dim2: $DIM2"
echo "npu_num: $NPU_NUM"
echo "dev_mode: $DEV_MODE"
echo "topology_file: $topology_file"
echo "workload_path: $workload_path"
echo "gpu_bootstrap: $GPU_COMPILE_BOOTSTRAP"
echo "init_tasks_diag_stage: $GPU_INIT_TASKS_DIAG_STAGE"
echo "gpu_device_heap_size: $GPU_DEVICE_HEAP_SIZE"
echo "================"

# ============================================================
# 自动计算 CHAKRA_NODES_DATA_LENGTH（如未手动指定）
# ============================================================
if [ -z "$CHAKRA_NODES_DATA_LENGTH" ]; then
    CHAKRA_NODES_DATA_LENGTH=$(cd "$ROOT" && PYTHONPATH="$ROOT/src" python3 \
        scripts/compute_chakra_capacity.py "$workload_path" --raw)
fi
echo "chakra_nodes_data_length: $CHAKRA_NODES_DATA_LENGTH"

if [ -z "$MAX_CHAKRA_NODES_PER_NPU" ]; then
    MAX_CHAKRA_NODES_PER_NPU=$(cd "$ROOT" && PYTHONPATH="$ROOT/src" python3 \
        scripts/compute_chakra_capacity.py "$workload_path" --raw-nodes)
fi
echo "max_chakra_nodes_per_npu: $MAX_CHAKRA_NODES_PER_NPU"

# ============================================================
# 替换文件中的值
# ============================================================
SYS_TYPES="$ROOT/src/sys/sys_types.hpp"
SYS_CONFIG="$ROOT/src/sys/sys_config.hpp"
CHAKRA_CONFIG="$ROOT/src/madrona_simple_example/chakra/config.py"

# 替换 C++ 编译期 NPU 上限（也是 Chakra 数据 tensor 的行数）
if [ -f "$SYS_TYPES" ]; then
    sed -i "s/^#define[[:space:]]*NPU_NUM[[:space:]]*[0-9]*/#define NPU_NUM $NPU_NUM/" "$SYS_TYPES"
    sed -i "s/^#define[[:space:]]*CHAKRA_NODES_DATA_LENGTH[[:space:]]*[0-9]*/#define CHAKRA_NODES_DATA_LENGTH $CHAKRA_NODES_DATA_LENGTH/" "$SYS_TYPES"
    sed -i "s/^#define[[:space:]]*MAX_CHAKRA_NODES_PER_NPU[[:space:]]*[0-9]*/#define MAX_CHAKRA_NODES_PER_NPU $MAX_CHAKRA_NODES_PER_NPU/" "$SYS_TYPES"
    echo "✓ 已更新 $SYS_TYPES"
else
    echo "警告: 未找到 $SYS_TYPES"
fi

# 替换 DEV_MODE 与日志开关
if [ -f "$SYS_CONFIG" ]; then
    sed -i "s/^#define[[:space:]]*DEV_MODE[[:space:]]*[0-9]*/#define DEV_MODE $DEV_MODE/" "$SYS_CONFIG"
    sed -i "s/^#define[[:space:]]*SIMPLE_LOG_MODE[[:space:]]*[0-9]*/#define SIMPLE_LOG_MODE $SIMPLE_LOG_MODE/" "$SYS_CONFIG"
    sed -i "s/^#define[[:space:]]*SYS_LOG[[:space:]][0-9]*/#define SYS_LOG $SYS_LOG/" "$SYS_CONFIG"
    sed -i "s/^#define[[:space:]]*SYS_LOG_SPECIAL[[:space:]]*[0-9]*/#define SYS_LOG_SPECIAL $SYS_LOG_SPECIAL/" "$SYS_CONFIG"
    sed -i "s/^#define[[:space:]]*SYS_LOG_TARGET_NODE[[:space:]]*[0-9]*/#define SYS_LOG_TARGET_NODE $SYS_LOG_TARGET_NODE/" "$SYS_CONFIG"
    echo "✓ 已更新 $SYS_CONFIG"
else
    echo "警告: 未找到 $SYS_CONFIG"
fi

# 替换 Python 端 NPU 上限 / 数据长度，必须与 C++ 一致（否则张量形状不匹配）
if [ -f "$CHAKRA_CONFIG" ]; then
    sed -i "s/^MAX_NPUS[[:space:]]*=[[:space:]]*[0-9]*/MAX_NPUS = $NPU_NUM/" "$CHAKRA_CONFIG"
    sed -i "s/^CHAKRA_NODES_DATA_LENGTH[[:space:]]*=[[:space:]]*[0-9]*/CHAKRA_NODES_DATA_LENGTH = $CHAKRA_NODES_DATA_LENGTH/" "$CHAKRA_CONFIG"
    echo "✓ 已更新 $CHAKRA_CONFIG"
else
    echo "警告: 未找到 $CHAKRA_CONFIG"
fi

echo ""

# ============================================================
# GPU 编译环境变量（缓存文件名带指纹，改参数/代码后自动失效）
# ============================================================
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "错误: 未找到 CUDA 12.8: $CUDA_HOME" >&2
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/lib"
export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p "$ROOT/.cache"
cuda_version=$("$CUDA_HOME/bin/nvcc" --version |
    sed -n 's/.*release \([0-9.]*\).*/\1/p')
source_hash=$(
    {
        git -C "$ROOT" diff --no-ext-diff HEAD -- src 2>/dev/null
        sha256sum "$ROOT/external/madrona/src/mw/cuda_exec.cpp"
        sha256sum "$ROOT/external/madrona/src/mw/device/include/madrona/mw_gpu_entry.hpp"
    } | sha256sum | cut -c1-12
)
compile_key="cuda${cuda_version}_src${source_hash}_npu${NPU_NUM}_d${CHAKRA_NODES_DATA_LENGTH}_m${MAX_CHAKRA_NODES_PER_NPU}_dev${DEV_MODE}_boot${GPU_COMPILE_BOOTSTRAP}_it${GPU_INIT_TASKS_DIAG_STAGE}_$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo head)"
export MADRONA_MWGPU_KERNEL_CACHE="$ROOT/.cache/mw_megakernel_${compile_key}.bin"
export MADRONA_BVH_KERNEL_CACHE="$ROOT/.cache/mw_bvh_${compile_key}.bin"
export MADRONA_MWGPU_VERBOSE_COMPILE=1
export MADRONA_MWGPU_TRACE_INIT=$GPU_TRACE_INIT
export MADRONA_MWGPU_INIT_TASKS_DIAG_STAGE=$GPU_INIT_TASKS_DIAG_STAGE
export MADRONA_MWGPU_DEVICE_HEAP_SIZE=$GPU_DEVICE_HEAP_SIZE
if [ "$GPU_COMPILE_BOOTSTRAP" = 1 ]; then
    export MADRONA_MWGPU_FORCE_DEBUG=1
    export MADRONA_MWGPU_OPT_MODE=debug
else
    unset MADRONA_MWGPU_FORCE_DEBUG
    # CUDA 12.8 NVRTC generates invalid LLVM debug metadata for Madrona's
    # cuda::atomic code when device LTO is enabled. Optimize mode keeps
    # device optimization but emits CUBIN per source instead of LTO IR.
    export MADRONA_MWGPU_OPT_MODE=optimize
fi
echo "cuda_home: $CUDA_HOME (CUDA $cuda_version)"
echo "gpu_opt_mode: $MADRONA_MWGPU_OPT_MODE"
echo "kernel_cache: $MADRONA_MWGPU_KERNEL_CACHE"

# 流仿真器网络日志：设置环境变量（system_log_print_enabled=1 同时启用 trace 模式）
if [ "$NET_LOG" = 1 ]; then
    export system_log_print_enabled=1
    [ -n "$NET_LOG_SCOPE" ] && export system_log_scope="$NET_LOG_SCOPE"
    [ -n "$NET_LOG_EVERY" ] && export system_log_every="$NET_LOG_EVERY"
fi

# ============================================================
# 构建
# ============================================================
echo "=== building flow simulator (CPU) ==="
cmake --build "$ROOT/build" -j$(nproc)

# ============================================================
# 运行（GPU）
# ============================================================
ARGS=(--topo "$topology_file" --workload "$workload_path")
ARGS+=(--npu-count "$NPU_NUM" --ring-dims "$DIM0,$DIM1,$DIM2")
ARGS+=(--max-steps "$max_steps" --print-every "$PRINT_EVERY" --gpu)
[ -n "$out_csv" ] && ARGS+=(--out-csv "$out_csv")

echo "=== running flow simulator (GPU) ==="
cd "$ROOT"
if [ -n "$run_log" ]; then
    echo "日志写入: $run_log"
    mkdir -p "$(dirname "$run_log")"
    PYTHONPATH="$ROOT/src" \
        python3 scripts/run_flow_simulator.py "${ARGS[@]}" > "$run_log" 2>&1
else
    PYTHONPATH="$ROOT/src" \
        python3 scripts/run_flow_simulator.py "${ARGS[@]}"
fi
