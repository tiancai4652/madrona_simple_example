"""冒烟探针：1P+1D 闭合 workload_pd32，小步验证 Prefill 能完成、KV 开始（不要求 finish）。

用法：python3 docs/runs/002-DeepSeekV3-671B训练负载/probe_pd32.py [N] [trace]
默认 N=5000，trace=request_trace_smoke.json（1x512x1，秒级验证链路）。
全量 trace 用 request_trace.json（4x4096x8）时建议 N>=20000。
判据：任一请求 p_finish_ns>0（Prefill 完成）且 kv_start_ns>0（KV 开始），无 FAILED。
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd() / "src"))

from madrona_simple_example import GridWorld, InferenceConfig, SystemConfig
from madrona_simple_example.chakra.conversion import folder_to_int_array
from madrona_simple_example.gridworld import load_network_inputs_from_files

ROUND = Path("docs/runs/002-DeepSeekV3-671B训练负载")
TRACE = sys.argv[2] if len(sys.argv) > 2 else str(ROUND / "request_trace_smoke.json")
PROFILE = str(ROUND / "op_profile.json")
TOPO = "/app/exps/inputs/topo/n32_4leaf/leafspine_h32_topo.txt"
WORKLOAD = str(ROUND / "workload_pd32")

rows = folder_to_int_array(Path(WORKLOAD), max_workers=0, show_progress=False)
npu_count = len(rows)
assert npu_count == 32, npu_count
world = GridWorld(
    num_worlds=1,
    start_cell=np.array([0, 0]),
    end_cells=np.array([[0, 0]], dtype=np.int32),
    rewards=np.zeros((1, 1), dtype=np.float32),
    walls=np.zeros((1, 1), dtype=bool),
    gpu_sim=False,
    network_inputs=load_network_inputs_from_files(TOPO, None),
    system_workload=rows,
    system_config=SystemConfig(ring_dims=(32, 1, 1), chunks_num=1, npu_count=32),
    inference_config=InferenceConfig(
        enabled=True, prefill_workers=(tuple(range(0, 16)),),
        decode_workers=(tuple(range(16, 32)),),
        p_max_batch=16, p_max_tokens=8192, d_max_batch=16, d_max_tokens=32768,
        num_layers=61, num_kv_heads=128, head_dim=128, bytes_per_elem=2,
        kv_partition_factor=0, prefill_reference_tokens=4096,
        decode_reference_tokens=4096, kv_mode=1, kv_latent_dim=576),
    request_trace=TRACE, workload_params_table=PROFILE)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
KEYS = ("request_id", "state", "p_worker", "d_worker", "p_start_ns",
        "p_finish_ns", "kv_start_ns", "kv_done_ns", "kv_bytes",
        "kv_shards", "kv_shards_done", "d_start_ns", "output_done",
        "output_len", "first_token_ns", "finish_ns")
checkpoints = sorted({100, 300, 1000, 3000, 5000, 10000, 20000, N} & set(range(1, N + 1)))
if N not in checkpoints:
    checkpoints.append(N)
for step in range(1, N + 1):
    world.step()
    if step in checkpoints:
        print(f"--- step={step} sim_ms={world.simulation_time():.4f} status={world.system_status()}", flush=True)
        for r in world.inference_stats():
            print("   ", {k: r[k] for k in KEYS}, flush=True)
status = world.system_status()
stats = world.inference_stats()
p_done = sum(1 for r in stats if r["p_finish_ns"] > 0)
kv_started = sum(1 for r in stats if r["kv_start_ns"] > 0)
finished = sum(1 for r in stats if r["finish_ns"] > 0)
print(f"[probe] steps={N} failed={status['failed']} "
      f"prefill_done={p_done}/{len(stats)} kv_started={kv_started}/{len(stats)} "
      f"requests_finished={finished}/{len(stats)}")
