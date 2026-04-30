#!/usr/bin/env bash
# 跑 leafspine1024_lowdelay alltoall 场景，并打开每步 workload 统计打印。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MADRONA_RUNNER="$ROOT/madrona_simple_example/scripts/run_parity.py"
TOPO="$ROOT/jiuding_dodsim/examples/leafspine1024_lowdelay/leafspine_h1024_topo_oversub.txt"
FLOWS="$ROOT/jiuding_dodsim/examples/leafspine1024_lowdelay/leafspine_h1024_d64_alltoall_2m.txt"
OUT_DIR="${1:-$ROOT/exp/out/madrona_leafspine1024_stepwork-cpu}"

MAX_STEPS="${MAX_STEPS:-200000}"
NUM_WORLDS="${NUM_WORLDS:-1}"
BACKEND="${MADRONA_BACKEND:-cpu}"

if [[ ! -f "$MADRONA_RUNNER" ]]; then
  echo "Missing madrona runner: $MADRONA_RUNNER" >&2
  exit 1
fi
if [[ ! -f "$TOPO" ]]; then
  echo "Missing topo file: $TOPO" >&2
  exit 1
fi
if [[ ! -f "$FLOWS" ]]; then
  echo "Missing flows file: $FLOWS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

RUN_LOG="$OUT_DIR/step_workload_run.log"
STEP_WORK_LOG="$OUT_DIR/step_workload_lines.txt"
STEP_WORK_CSV="$OUT_DIR/step_workload.csv"

export MADSIMPLE_STEP_WORKLOAD="${MADSIMPLE_STEP_WORKLOAD:-1}"
export MADSIMPLE_PRINT_STEP_WORKLOAD="${MADSIMPLE_PRINT_STEP_WORKLOAD:-1}"

GPU_FLAG=""
if [[ "$BACKEND" == "gpu" ]]; then
  export MADRONA_MWGPU_VERBOSE_COMPILE="${MADRONA_MWGPU_VERBOSE_COMPILE:-1}"
  export MADRONA_MWGPU_KERNEL_CACHE="${MADRONA_MWGPU_KERNEL_CACHE:-$ROOT/.cache/mw_megakernel_cache.bin}"
  export MADRONA_BVH_KERNEL_CACHE="${MADRONA_BVH_KERNEL_CACHE:-$ROOT/.cache/mw_bvh_cache.bin}"
  export MADRONA_MWGPU_OPT_MODE="${MADRONA_MWGPU_OPT_MODE:-optimize}"
  mkdir -p "$(dirname "$MADRONA_MWGPU_KERNEL_CACHE")"
  GPU_FLAG="--gpu"
fi

echo "===================================================="
echo "  leafspine1024 alltoall madrona step workload test"
echo "===================================================="
echo "topo:         $TOPO"
echo "flows:        $FLOWS"
echo "backend:      $BACKEND"
echo "max_steps:    $MAX_STEPS"
echo "num_worlds:   $NUM_WORLDS"
echo "out_dir:      $OUT_DIR"
echo "run_log:      $RUN_LOG"
echo "step_log:     $STEP_WORK_LOG"
echo "step_csv:     $STEP_WORK_CSV"
echo "step_work:    $MADSIMPLE_STEP_WORKLOAD"
echo "print_step:   $MADSIMPLE_PRINT_STEP_WORKLOAD"
if [[ "$BACKEND" == "gpu" ]]; then
  echo "kernel_cache: $MADRONA_MWGPU_KERNEL_CACHE"
  echo "bvh_cache:    $MADRONA_BVH_KERNEL_CACHE"
  echo "gpu_opt_mode: $MADRONA_MWGPU_OPT_MODE"
fi
echo ""

python3 "$MADRONA_RUNNER" \
  --topo "$TOPO" \
  --flows "$FLOWS" \
  --out-dir "$OUT_DIR" \
  --num-worlds "$NUM_WORLDS" \
  --max-steps "$MAX_STEPS" \
  $GPU_FLAG 2>&1 | tee "$RUN_LOG"

grep '^\[STEP_WORK\]' "$RUN_LOG" > "$STEP_WORK_LOG" || true

python3 - "$STEP_WORK_LOG" "$STEP_WORK_CSV" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

columns = [
    "step",
    "now",
    "ready_flows",
    "scheduled_events",
    "due_events",
    "due_arrival",
    "due_bwupdate",
    "due_pfc",
    "due_ports",
    "dirty_ports",
    "alloc_ports",
    "alloc_tags",
    "create_reqs",
    "cleanup_reqs",
    "completion_reqs",
    "outbox_events",
    "outbox_ports",
    "active_tags",
    "source_tags",
    "ingress_tags",
    "finished_sources",
    "emitted_cleanup",
    "drain_timers",
    "pause_timers",
    "resume_timers",
    "next_dt",
]

rows = []
if src.exists():
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("[STEP_WORK] "):
            continue

        rec = {key: "" for key in columns}
        for token in line[len("[STEP_WORK] "):].split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key in rec:
                rec[key] = value
        rows.append(rec)

with dst.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
PY

echo ""
echo "===================================================="
echo "  madrona step workload 运行完成，输出目录: $OUT_DIR"
echo "  完整日志: $RUN_LOG"
echo "  逐步统计: $STEP_WORK_LOG"
echo "  CSV统计:   $STEP_WORK_CSV"
echo "===================================================="
