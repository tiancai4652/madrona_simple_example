#!/bin/bash
# ============================================================
# GPU 验收脚本：一次跑完「Debug bootstrap 打通」+「正式 LTO 编译运行」
# 用法：   bash run_gpu_verify.sh > 1.log 2>&1
#          （所有输出只走一个流，全部进 1.log，不另产生日志文件）
# ============================================================
set -uo pipefail

ROOT=/app/madrona_simple_example
SCRIPT="$ROOT/run_flow_simulator.sh"
# 临时文件仅用于内部诊断（最后一行编译到哪个 .cpp），用完即删，不落盘
TMP_LOG=$(mktemp /tmp/gpu_run_XXXXXX.log)

echo "=== GPU verify start: $(date) ==="
nvidia-smi || { echo "FATAL: nvidia-smi 失败，没有可用 GPU"; exit 2; }

cd "$ROOT" || exit 2

# 0) 确保脚本里 gpu=1
sed -i 's/^gpu=0.*/gpu=1   # 0 = CPU, 1 = GPU/' "$SCRIPT"

# ============================================================
# STEP 1: Debug bootstrap（打通编译链路，快）
# ============================================================
sed -i 's/^GPU_COMPILE_BOOTSTRAP=0.*/GPU_COMPILE_BOOTSTRAP=1/' "$SCRIPT"
echo ""
echo "=== [1/2] Bootstrap(Debug) run: 打通 GPU 编译链路 ==="
MADRONA_MWGPU_VERBOSE_COMPILE=1 bash "$SCRIPT" 2>&1 | tee "$TMP_LOG"
boot_rc=${PIPESTATUS[0]}
echo "bootstrap exit code = $boot_rc"

echo ""
echo "=== 缓存检查 ==="
mkdir -p "$ROOT/.cache"
ls -la "$ROOT/.cache/"
boot_cache=$(ls "$ROOT/.cache/"/mw_megakernel_*boot1*.bin 2>/dev/null | head -1)
if [ -n "$boot_cache" ] && [ "$(stat -c%s "$boot_cache" 2>/dev/null)" -gt 0 ]; then
    echo "OK: bootstrap 缓存已落盘 ($boot_cache, $(stat -c%s "$boot_cache") bytes)"
else
    echo "WARN: bootstrap 缓存缺失或为空 —— 编译未成功"
fi

echo ""
echo "=== 编译卡点定位（最后一个被编译的 .cpp）==="
grep -oE 'src/[^"]*\.cpp' "$TMP_LOG" | tail -1 || true

# ============================================================
# STEP 2: 正式 LTO（冷编译 + 运行）
# ============================================================
sed -i 's/^GPU_COMPILE_BOOTSTRAP=1.*/GPU_COMPILE_BOOTSTRAP=0/' "$SCRIPT"
echo ""
echo "=== [2/2] LTO run: 正式冷编译 + 运行 ==="
echo "       （这一步最慢：预计 10~60 分钟。看 1.log 里'Compiling GPU engine code:'"
echo "         之后长时间无新 .cpp 打印，说明某个文件在卡）"
MADRONA_MWGPU_VERBOSE_COMPILE=1 bash "$SCRIPT" 2>&1 | tee -a "$TMP_LOG"
lto_rc=${PIPESTATUS[0]}
echo "LTO exit code = $lto_rc"

# ============================================================
# STEP 3: 汇总
# ============================================================
echo ""
echo "=== [3/3] 汇总 ==="
echo "bootstrap: $([ $boot_rc -eq 0 ] && echo PASS || echo FAIL)"
echo "LTO:       $([ $lto_rc -eq 0 ] && echo PASS || echo FAIL)"

lto_cache=$(ls "$ROOT/.cache/"/mw_megakernel_*boot0*.bin 2>/dev/null | head -1)
if [ -n "$lto_cache" ] && [ "$(stat -c%s "$lto_cache" 2>/dev/null)" -gt 0 ]; then
    echo "LTO 缓存已落盘: $lto_cache ($(stat -c%s "$lto_cache") bytes)"
    echo "下次同参数运行将秒级加载缓存，无需再编译。"
fi

if [ $boot_rc -eq 0 ] && [ $lto_rc -eq 0 ]; then
    echo "ALL PASS —— GPU 编译问题已解决"
else
    echo "存在失败，见上面的输出。若卡在某个 .cpp，把 1.log 里最后一个 src/xxx.cpp 发给我。"
fi

rm -f "$TMP_LOG"
exit $((boot_rc || lto_rc))
