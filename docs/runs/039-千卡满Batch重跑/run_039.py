"""039-千卡满Batch重跑 runner (单世界, t2=5ms固定, 复用已验证workload).

用法:
  python3 run_039.py full <e1|e2|e3> <gpu_id>

E1 100-1024 deepseek layer0: 1024卡4P+4D, B128, Lin4096 Lout2, 512+512@0/5ms, p524288/d524544, MLA61
E2 1024 deepseek layer0:     1024卡4P+4D, B128, Lin4096 Lout4, 512+512@0/5ms, p524288/d524800, MLA61
E3 1024 llama405B tp64:      1024卡4P+4D, B128, Lin4096 Lout4, 512+512@0/5ms, p524288/d524800, GQA126
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from madrona_simple_example import GridWorld, SystemConfig
from madrona_simple_example.chakra.conversion import folder_to_int_array
from madrona_simple_example.gridworld import load_network_inputs_from_files
from madrona_simple_example.inference.config import InferenceConfig

RUN_DIR = Path(__file__).resolve().parent
TOPO = Path("/app/华为2期交付/1024/deepseekv3-671B/leafspine_h1024_topo_oversub2.txt")

PROMPT = 4096
N_NPUS = 1024
T2_NS = 5_000_000

P_WORKERS = tuple(tuple(range(i * 128, (i + 1) * 128)) for i in range(4))
D_WORKERS = tuple(tuple(range(512 + i * 128, 512 + (i + 1) * 128)) for i in range(4))

SPECS = {
    "e1": {"WL": Path("/app/madrona_simple_example/docs/runs/036-需求x单实验与共享/workload_036_layer0_sc_rows.npy"),
           "N": 512, "OUT": 2, "B": 128, "model": "ds",
           "label": "100-1024-deepseek-layer0"},
    "e2": {"WL": Path("/app/华为2期交付/1024/deepseekv3-671B/exp/workload_1024_fanout_layer0_sc"),
           "N": 512, "OUT": 4, "B": 128, "model": "ds",
           "label": "1024-deepseek-layer0"},
    "e3": {"WL": Path("/app/华为2期交付/1024/lama3-405B/exp/workload_1024_fanout_tp64"),
           "N": 512, "OUT": 4, "B": 128, "model": "llama",
           "label": "1024-llama-tp64"},
}


def make_ic(spec):
    B, OUT = spec["B"], spec["OUT"]
    if spec["model"] == "ds":
        return InferenceConfig(
            enabled=True, prefill_workers=P_WORKERS, decode_workers=D_WORKERS,
            p_max_batch=B, p_max_tokens=B * PROMPT,
            d_max_batch=B, d_max_tokens=B * (PROMPT + OUT),
            num_layers=61, num_kv_heads=128, head_dim=128, bytes_per_elem=2,
            kv_partition_factor=0,
            prefill_reference_tokens=B * PROMPT,
            decode_reference_tokens=B * (PROMPT + OUT),
            kv_mode=1, kv_latent_dim=576)
    return InferenceConfig(
        enabled=True, prefill_workers=P_WORKERS, decode_workers=D_WORKERS,
        p_max_batch=B, p_max_tokens=B * PROMPT,
        d_max_batch=B, d_max_tokens=B * (PROMPT + OUT),
        num_layers=126, num_kv_heads=8, head_dim=128, bytes_per_elem=2,
        kv_partition_factor=0,
        prefill_reference_tokens=B * PROMPT,
        decode_reference_tokens=B * (PROMPT + OUT),
        kv_mode=0, kv_latent_dim=0)


def req(i, arrival_ns, out):
    return {"request_id": i, "arrival_time_ns": arrival_ns,
            "prompt_len": PROMPT, "output_len": out}


def run_full(tag, spec, gpu_id, max_steps=200000):
    B, OUT, N, WL = spec["B"], spec["OUT"], spec["N"], spec["WL"]
    mem_samples, sim_mem_samples = [], []
    in_sim, stop_flag = False, False

    def sampler():
        while not stop_flag:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits", "-i", str(gpu_id)]).decode().strip()
                mem_samples.append(int(out.splitlines()[0].strip()))
                if in_sim:
                    sim_mem_samples.append(mem_samples[-1])
            except Exception:
                pass
            time.sleep(3)

    t_smi = threading.Thread(target=sampler, daemon=True)
    t_smi.start()
    try:
        ic = make_ic(spec)
        ic.validate(npu_count=N_NPUS)
        print(f"[{tag}] B={B} p_max={B*PROMPT} d_max={B*(PROMPT+OUT)} model={spec['model']}", flush=True)
        t_start = time.time()
        topo = load_network_inputs_from_files(str(TOPO), None)
        print(f"[{tag}] topo nodes={len(topo['node_ids'])} links={len(topo['link_srcs'])} wall={time.time()-t_start:.2f}s", flush=True)
        rows = (np.load(str(WL)) if str(WL).endswith(".npy")
                else folder_to_int_array(str(WL), max_workers=16, show_progress=False))
        print(f"[{tag}] encode files={len(rows)} wall={time.time()-t_start:.1f}s", flush=True)
        trace = [req(i, 0, OUT) for i in range(N)] + [req(N + i, T2_NS, OUT) for i in range(N)]
        tp = RUN_DIR / f"request_trace_039_{tag}.json"
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=1)
        print(f"[{tag}] trace n={len(trace)} t2=5ms -> {tp}", flush=True)
        world = GridWorld(
            system_workload=rows,
            system_config=SystemConfig(ring_dims=(N_NPUS, 1, 1), chunks_num=1, npu_count=N_NPUS),
            network_inputs=topo, inference_config=ic, request_trace=str(tp),
            workload_params_table=None, num_worlds=1,
            start_cell=np.array([0, 0]),
            end_cells=np.array([[0, 0]], dtype=np.int32),
            rewards=np.zeros((1, 1), dtype=np.float32),
            walls=np.zeros((1, 1), dtype=bool),
            gpu_sim=True, gpu_id=gpu_id)
        t_init = time.time()
        in_sim = True
        try:
            init_mem = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", str(gpu_id)]).decode().strip().splitlines()[0])
            mem_samples.append(init_mem)
            sim_mem_samples.append(init_mem)
        except Exception:
            init_mem = mem_samples[-1] if mem_samples else 0
        print(f"[{tag}] init OK init_wall={t_init-t_start:.2f}s gpu{gpu_id}_mem={init_mem}MiB", flush=True)
        finished_step = 0
        for step in range(1, max_steps + 1):
            world.step()
            if step == 1 or step % 10 == 0 or step == max_steps:
                st = world.system_status(0)
                cur = sim_mem_samples[-1] if sim_mem_samples else (mem_samples[-1] if mem_samples else 0)
                print(f"[{tag}] step={step} sim_ms={world.simulation_time(0):.4f} finished={st.get('finished')} "
                      f"err={st.get('error_code')} gpu_mem={cur}MiB wall={time.time()-t_start:.1f}s sim_wall={time.time()-t_init:.1f}s", flush=True)
            st = world.system_status(0)
            assert not st.get("failed"), st
            if st.get("finished"):
                finished_step = step
                break
        t_end = time.time()
        in_sim = False
        st = world.system_status(0)
        sim_ms = world.simulation_time(0)
        print(f"[{tag}] steps_run={finished_step or max_steps} finished={st.get('finished')} err={st.get('error_code')} "
              f"sim_ms={sim_ms:.6f} total_wall={t_end-t_start:.2f}s sim_wall={t_end-t_init:.2f}s "
              f"step_s={(t_end-t_init)/(finished_step or max_steps):.4f} pmax={world.port_max_levels(0)}", flush=True)
        if finished_step:
            print(f"[{tag}] RUN OK steps={finished_step}", flush=True)
            out_csv = RUN_DIR / f"inference_stats_039_{tag}_gpu{gpu_id}.csv"
            world.write_inference_stats_csv(str(out_csv), 0)
            print(f"[{tag}] wrote {out_csv}", flush=True)
            stats = world.inference_stats(0)
            fin = [r["finish_ns"] / 1e6 for r in stats]
            pfin = [r["p_finish_ns"] / 1e6 for r in stats]
            states = [r["state"] for r in stats]
            print(f"[{tag}] n={len(stats)} all7={all(s == 7 for s in states)} "
                  f"finish_max={max(fin):.6f} finish_min={min(fin):.6f} "
                  f"p_finish_max={max(pfin):.6f} p_finish_min={min(pfin):.6f}", flush=True)
    finally:
        stop_flag = True
        t_smi.join(timeout=4)
        with open(RUN_DIR / f"smi_039_{tag}.log", "w", encoding="utf-8") as f:
            for v in mem_samples:
                f.write(f"{v}\n")
        print(f"[{tag}] SMI peak_all={max(mem_samples) if mem_samples else 0}MiB "
              f"peak_sim={max(sim_mem_samples) if sim_mem_samples else 0}MiB n={len(mem_samples)}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    e = sys.argv[2] if len(sys.argv) > 2 else "e1"
    gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    assert e in SPECS, e
    run_full(e, SPECS[e], gpu)
