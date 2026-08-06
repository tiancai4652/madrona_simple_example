#!/bin/bash

# Run the merged flow simulator end-to-end, modeled on multiverse-dev/run.sh
# (config params + source-constant replacement + build + run).

set -euo pipefail

ROOT=/app/madrona_simple_example

# ============================================================
# 配置参数：直接在这里修改参数值
# ============================================================
DIM0=8
DIM1=2
DIM2=8
NPU_NUM=128
DEV_MODE=2   # 0 = SYS, 1 = NET, 2 = MIX

# 日志开关（对应 multiverse global_types.hpp；SIMPLE_LOG_MODE 与 SYS_LOG 二选一）
SIMPLE_LOG_MODE=0   # 1 = 简单日志（需 SYS_LOG=0）
SYS_LOG=1           # 1 = 详细日志（需 SIMPLE_LOG_MODE=0）
SYS_LOG_SPECIAL=0
SYS_LOG_TARGET_NODE=0   # 只打印该 NPU 的日志

# 流仿真器网络日志（std::cout 的 [SYS][...] 输出；开启会自动启用 trace 模式）
NET_LOG=1               # 1 = 开启流仿真器网络日志
NET_LOG_SCOPE=all       # all 或 ingress_chain / emit_tag
NET_LOG_EVERY=10        # 每 N 步打印一次，控制日志量

# 运行日志输出文件（留空则不落盘，直接打印到终端）
run_log=/app/report/2-merge-plan/实施/run.log
topology_file=/app/jiuding_dodsim/examples/leafspine128/leafspine_h128_topo.txt
workload_path=/app/multiverse-dev/scripts/input/workload/128gpus-tp8-pp2-dp8-vpp2-gbs32-mbs1-hid_size12288-seq_len2048-json-comm_scale100-comp_scale100
# workload_path=/app/multiverse-dev/scripts/input/workload/16_4_2_2
max_steps=200000
gpu=0   # 0 = CPU, 1 = GPU
out_csv=
# CHAKRA_NODES_DATA_LENGTH 为空时按 workload 自动计算；也可手动指定
CHAKRA_NODES_DATA_LENGTH=
# MAX_CHAKRA_NODES_PER_NPU 为空时按 workload 最大节点数自动计算；也可手动指定
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
    echo "✓ 已更新 $SYS_TYPES"
else
    echo "警告: 未找到 $SYS_TYPES"
fi

# 替换 C++ 每 NPU 编码数据长度上限
if [ -f "$SYS_TYPES" ]; then
    sed -i "s/^#define[[:space:]]*CHAKRA_NODES_DATA_LENGTH[[:space:]]*[0-9]*/#define CHAKRA_NODES_DATA_LENGTH $CHAKRA_NODES_DATA_LENGTH/" "$SYS_TYPES"
    echo "✓ 已更新 $SYS_TYPES (CHAKRA_NODES_DATA_LENGTH)"
else
    echo "警告: 未找到 $SYS_TYPES"
fi

# 替换 C++ 每 NPU 最大 Chakra 节点数
if [ -f "$SYS_TYPES" ]; then
    sed -i "s/^#define[[:space:]]*MAX_CHAKRA_NODES_PER_NPU[[:space:]]*[0-9]*/#define MAX_CHAKRA_NODES_PER_NPU $MAX_CHAKRA_NODES_PER_NPU/" "$SYS_TYPES"
    echo "✓ 已更新 $SYS_TYPES (MAX_CHAKRA_NODES_PER_NPU)"
else
    echo "警告: 未找到 $SYS_TYPES"
fi

# 替换 DEV_MODE（0=SYS, 1=NET, 2=MIX）
if [ -f "$SYS_CONFIG" ]; then
    sed -i "s/^#define[[:space:]]*DEV_MODE[[:space:]]*[0-9]*/#define DEV_MODE $DEV_MODE/" "$SYS_CONFIG"
    echo "✓ 已更新 $SYS_CONFIG"
else
    echo "警告: 未找到 $SYS_CONFIG"
fi

# 替换日志开关（SIMPLE_LOG_MODE / SYS_LOG / SYS_LOG_SPECIAL / SYS_LOG_TARGET_NODE）
if [ -f "$SYS_CONFIG" ]; then
    sed -i "s/^#define[[:space:]]*SIMPLE_LOG_MODE[[:space:]]*[0-9]*/#define SIMPLE_LOG_MODE $SIMPLE_LOG_MODE/" "$SYS_CONFIG"
    sed -i "s/^#define[[:space:]]*SYS_LOG[[:space:]][0-9]*/#define SYS_LOG $SYS_LOG/" "$SYS_CONFIG"
    sed -i "s/^#define[[:space:]]*SYS_LOG_SPECIAL[[:space:]]*[0-9]*/#define SYS_LOG_SPECIAL $SYS_LOG_SPECIAL/" "$SYS_CONFIG"
    sed -i "s/^#define[[:space:]]*SYS_LOG_TARGET_NODE[[:space:]]*[0-9]*/#define SYS_LOG_TARGET_NODE $SYS_LOG_TARGET_NODE/" "$SYS_CONFIG"
    echo "✓ 已更新 $SYS_CONFIG (日志开关: SIMPLE_LOG_MODE=$SIMPLE_LOG_MODE SYS_LOG=$SYS_LOG SYS_LOG_SPECIAL=$SYS_LOG_SPECIAL SYS_LOG_TARGET_NODE=$SYS_LOG_TARGET_NODE)"
else
    echo "警告: 未找到 $SYS_CONFIG"
fi

# 替换 Python 端 NPU 上限，必须与 C++ NPU_NUM 一致（否则张量形状不匹配）
if [ -f "$CHAKRA_CONFIG" ]; then
    sed -i "s/^MAX_NPUS[[:space:]]*=[[:space:]]*[0-9]*/MAX_NPUS = $NPU_NUM/" "$CHAKRA_CONFIG"
    echo "✓ 已更新 $CHAKRA_CONFIG"
else
    echo "警告: 未找到 $CHAKRA_CONFIG"
fi

# 替换 Python 端每 NPU 编码数据长度，必须与 C++ CHAKRA_NODES_DATA_LENGTH 一致
if [ -f "$CHAKRA_CONFIG" ]; then
    sed -i "s/^CHAKRA_NODES_DATA_LENGTH[[:space:]]*=[[:space:]]*[0-9]*/CHAKRA_NODES_DATA_LENGTH = $CHAKRA_NODES_DATA_LENGTH/" "$CHAKRA_CONFIG"
    echo "✓ 已更新 $CHAKRA_CONFIG (CHAKRA_NODES_DATA_LENGTH)"
else
    echo "警告: 未找到 $CHAKRA_CONFIG"
fi

echo ""

# ============================================================
# 构建
# ============================================================
echo "=== building flow simulator (CPU) ==="
cmake --build "$ROOT/build" -j$(nproc)

# ============================================================
# 运行
# ============================================================
ARGS=(--topo "$topology_file" --workload "$workload_path")
ARGS+=(--npu-count "$NPU_NUM" --ring-dims "$DIM0,$DIM1,$DIM2")
ARGS+=(--max-steps "$max_steps")
[ "$gpu" = 1 ]    && ARGS+=(--gpu)
[ -n "$out_csv" ] && ARGS+=(--out-csv "$out_csv")

# 流仿真器网络日志：设置环境变量（system_log_print_enabled=1 同时启用 trace 模式）
if [ "$NET_LOG" = 1 ]; then
    export system_log_print_enabled=1
    [ -n "$NET_LOG_SCOPE" ] && export system_log_scope="$NET_LOG_SCOPE"
    [ -n "$NET_LOG_EVERY" ] && export system_log_every="$NET_LOG_EVERY"
fi

echo "=== running flow simulator ==="
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
