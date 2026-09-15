"""Frame-by-frame trace of the B1-milestone inference simulation demo.

Reuses the scenario from run_serving_complex_example.py and adds:
  - per-step narration (sim time, state transitions, KV flows, generations)
  - machine-readable trace CSV next to the run output
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "complex_example", HERE / "run_serving_complex_example.py")
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)

STATE_NAMES = {
    1: "待到达", 2: "排队prefill", 3: "prefill中",
    4: "KV传输中", 5: "排队decode", 6: "decode中", 7: "完成",
}
KV_BIT = 0x80000000


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/opencode/infer_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    world = example.make_world()
    prev_states = {}
    prev_kv_done = 0
    prev_tokens = 0
    events = []
    trace_rows = []

    for step in range(1, 2001):
        world.step()
        status = world.system_status()
        sim_ns = int(round(world.simulation_time() * 1e6))
        rows = world.inference_stats()
        step_events = []

        for r in rows:
            rid = r["request_id"]
            st = r["state"]
            old = prev_states.get(rid, 1)
            if st != old:
                step_events.append(
                    f"req{rid}: {STATE_NAMES.get(old, old)}->{STATE_NAMES.get(st, st)}"
                    f" (P{r['p_worker']}/D{r['d_worker']}"
                    f" out={r['output_done']}/{r['output_len']})")
            prev_states[rid] = st

        kv_done = sum(1 for f in world.flow_completions() if f["flow_id"] & KV_BIT)
        if kv_done > prev_kv_done:
            step_events.append(f"KV流完成 +{kv_done - prev_kv_done} (累计 {kv_done})")
        prev_kv_done = kv_done

        tokens = sum(r["output_done"] for r in world.inference_stats())
        if tokens > prev_tokens:
            step_events.append(f"生成token +{tokens - prev_tokens} (累计 {tokens})")
        prev_tokens = tokens

        trace_rows.append({
            "step": step, "sim_time_ns": sim_ns,
            "delayed_events": world.num_delayed_events(),
            "active_tags": world.num_active_tags(),
            "events": "; ".join(step_events),
        })
        if step_events or step <= 3:
            events.append((step, sim_ns, step_events))

        if status["failed"]:
            raise RuntimeError(f"failed at step {step}: {status}")
        if status["finished"]:
            break

    with (out_dir / "frame_trace.csv").open("w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["step", "sim_time_ns", "delayed_events",
                                          "active_tags", "events"])
        w.writeheader()
        w.writerows(trace_rows)

    final_rows = world.inference_stats()
    with (out_dir / "inference_stats.json").open("w") as f:
        json.dump(final_rows, f, indent=2, default=str)

    print(f"finished in {step} steps, sim_time_ns={sim_ns}")
    print(f"{'step':>5} {'sim_ns':>9}  events")
    for s, t, evs in events:
        print(f"{s:>5} {t:>9}  {'; '.join(evs)}")
    print("\nfinal stats:")
    print("request P D prompt out kvB p_start p_fin kv_done 1st_tok fin")
    for r in final_rows:
        print(f"{r['request_id']:>7} {r['p_worker']} {r['d_worker']} "
              f"{r['prompt_len']:>6} {r['output_done']:>3} {r['kv_bytes']:>5} "
              f"{r['p_start_ns']:>6} {r['p_finish_ns']:>6} {r['kv_done_ns']:>7} "
              f"{r['first_token_ns']:>7} {r['finish_ns']:>6}")


if __name__ == "__main__":
    main()
