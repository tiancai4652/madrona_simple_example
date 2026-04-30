import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "madrona_simple_example" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from madrona_simple_example import GridWorld, load_network_inputs_from_files


def build_grid_inputs():
    walls = np.zeros((1, 1), dtype=np.bool_)
    rewards = np.zeros((1, 1), dtype=np.float32)
    end_cells = np.array([[0, 0]], dtype=np.int32)
    start_cell = np.array([0, 0], dtype=np.int32)
    return walls, rewards, end_cells, start_cell


def parse_args():
    parser = argparse.ArgumentParser(description="Run Madrona parity scenario")
    parser.add_argument("--topo", required=True)
    parser.add_argument("--flows", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--t-max", type=float, default=1000.0)
    parser.add_argument("--prop-interval", type=float, default=0.0)
    parser.add_argument("--pfc", action="store_true")
    parser.add_argument("--pfc-egress", action="store_true")
    parser.add_argument("--pfc-xoff", type=float, default=1e9)
    parser.add_argument("--pfc-xon", type=float, default=0.5e9)
    parser.add_argument("--dt-min", type=float, default=0.0)
    parser.add_argument("--qos", type=str, default="none")
    parser.add_argument("--prior-weights", type=str, default="")
    return parser.parse_args()


def should_stop(world):
    return (
        world.num_pending_flows() == 0
        and world.num_delayed_events() == 0
        and world.num_active_tags() == 0
    )


def _is_log_capture_mode():
    """Return True when the run is generating raw [INIT]/[SYS] logs that will
    be parsed by check/run_parity.py. In that mode any [parity] trace from the
    Python side risks interleaving with C++ stdout (which is fully-buffered
    when stdout is a pipe) and breaking the parser."""
    for var in ("init_log_print_enabled", "system_log_print_enabled"):
        val = os.environ.get(var)
        if val not in (None, "", "0"):
            return True
    return False


def _pstage(msg, t_start):
    if _is_log_capture_mode():
        return
    print(f"[parity] {msg}  (+{time.time() - t_start:.2f}s)", flush=True)


def write_timing_summary(path: Path, summary: dict):
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    wall_clock_start = time.time()
    program_start = time.perf_counter()
    _pstage("main() entered", wall_clock_start)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_timing_path = out_dir / "madrona_step_times.csv"
    timing_summary_path = out_dir / "madrona_timing_summary.json"

    _pstage(
        f"loading inputs: topo={args.topo} flows={args.flows}",
        wall_clock_start,
    )
    load_inputs_start = time.perf_counter()
    network_inputs = load_network_inputs_from_files(Path(args.topo), Path(args.flows))
    walls, rewards, end_cells, start_cell = build_grid_inputs()
    load_inputs_elapsed = time.perf_counter() - load_inputs_start
    _pstage("inputs loaded", wall_clock_start)

    prior_weights = []
    if args.prior_weights:
        prior_weights = [float(x) for x in args.prior_weights.split(',')]

    gridworld_init_start = time.perf_counter()
    _pstage(
        f"constructing GridWorld(gpu={bool(args.gpu)}, pfc={bool(args.pfc)}, "
        f"num_worlds={args.num_worlds}) ...",
        wall_clock_start,
    )
    world = GridWorld(
        args.num_worlds,
        start_cell,
        end_cells,
        rewards,
        walls,
        gpu_sim=bool(args.gpu),
        gpu_id=1,
        network_inputs=network_inputs,
        propagation_interval=args.prop_interval,
        enable_pfc=1 if args.pfc else 0,
        pfc_egress=1 if args.pfc_egress else 0,
        pfc_xoff_threshold=args.pfc_xoff,
        pfc_xon_threshold=args.pfc_xon,
        dt_min=args.dt_min,
        qos_mode={"none": 0, "sp": 1, "wrr": 2}.get(args.qos, 0),
        prior_weights=prior_weights if prior_weights else None,
    )
    gridworld_init_elapsed = time.perf_counter() - gridworld_init_start
    _pstage(
        "GridWorld constructed (GPU init + launch graph built)",
        wall_clock_start,
    )

    init_elapsed = time.perf_counter() - program_start

    if os.environ.get("init_log_print_enabled") not in (None, "", "0"):
        write_timing_summary(
            timing_summary_path,
            {
                "gpu": bool(args.gpu),
                "init_input_load_wall_time_s": load_inputs_elapsed,
                "init_gridworld_wall_time_s": gridworld_init_elapsed,
                "init_wall_time_s": init_elapsed,
                "loop_wall_time_s": 0.0,
                "postprocess_wall_time_s": 0.0,
                "program_wall_time_s": time.perf_counter() - program_start,
                "steps_executed": 0,
                "step_wall_time_avg_ms": 0.0,
                "step_wall_time_min_ms": 0.0,
                "step_wall_time_max_ms": 0.0,
                "timing_mode": "init_only",
            },
        )
        return

    log_capture_mode = _is_log_capture_mode()
    progress_every = int(os.environ.get("PARITY_PROGRESS_EVERY", "500"))

    steps = 0
    loop_start_wall = time.time()
    loop_start = time.perf_counter()
    t_prev = loop_start_wall
    steps_prev = 0
    step_time_sum = 0.0
    step_time_min = None
    step_time_max = 0.0
    if not log_capture_mode:
        print(
            f"[parity] start loop: max_steps={args.max_steps} "
            f"gpu={bool(args.gpu)} pfc={bool(args.pfc)} "
            f"init_elapsed={init_elapsed:.3f}s "
            f"progress_every={progress_every}",
            flush=True,
        )

    with step_timing_path.open("w", newline="", encoding="utf-8", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "step_wall_time_s",
            "step_wall_time_ms",
            "cumulative_loop_wall_time_s",
        ])

        while steps < args.max_steps and not should_stop(world):
            step_start = time.perf_counter()
            world.step()
            step_elapsed = time.perf_counter() - step_start
            steps += 1

            step_time_sum += step_elapsed
            if step_time_min is None or step_elapsed < step_time_min:
                step_time_min = step_elapsed
            if step_elapsed > step_time_max:
                step_time_max = step_elapsed

            writer.writerow([
                steps,
                f"{step_elapsed:.9f}",
                f"{step_elapsed * 1e3:.6f}",
                f"{time.perf_counter() - loop_start:.9f}",
            ])

            if (
                not log_capture_mode
                and progress_every > 0
                and steps % progress_every == 0
            ):
                now = time.time()
                dt = max(now - t_prev, 1e-6)
                sps = (steps - steps_prev) / dt
                elapsed = now - loop_start_wall
                print(
                    f"[parity] step={steps}/{args.max_steps} "
                    f"sim_t={world.simulation_time():.3f} "
                    f"pend={world.num_pending_flows()} "
                    f"delayed={world.num_delayed_events()} "
                    f"active={world.num_active_tags()} "
                    f"step_ms={step_elapsed * 1e3:.3f} "
                    f"sps={sps:.1f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                t_prev = now
                steps_prev = steps

    loop_elapsed = time.perf_counter() - loop_start
    avg_step_ms = (step_time_sum / steps * 1e3) if steps > 0 else 0.0
    min_step_ms = (step_time_min * 1e3) if step_time_min is not None else 0.0
    max_step_ms = step_time_max * 1e3

    if not log_capture_mode:
        print(
            f"[parity] loop done: steps={steps} "
            f"elapsed={loop_elapsed:.3f}s "
            f"avg_step_ms={avg_step_ms:.3f} "
            f"min_step_ms={min_step_ms:.3f} "
            f"max_step_ms={max_step_ms:.3f} "
            f"stopped_cleanly={should_stop(world)}",
            flush=True,
        )

    postprocess_start = time.perf_counter()
    completion_path = out_dir / "flow_completion_times.csv"
    world.write_flow_completion_csv(completion_path)

    summary = {
        "steps_executed": steps,
        "max_steps": args.max_steps,
        "simulation_time": world.simulation_time(),
        "num_flow_defs": world.num_flow_defs(),
        "num_pending_flows": world.num_pending_flows(),
        "num_delayed_events": world.num_delayed_events(),
        "num_active_tags": world.num_active_tags(),
        "num_source_tags": world.num_source_tags(),
        "num_flow_completions": len(world.flow_completions()),
        "completion_csv": str(completion_path),
        "stopped_cleanly": should_stop(world),
    }

    summary_path = out_dir / "madrona_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    timing_summary = {
        "gpu": bool(args.gpu),
        "init_input_load_wall_time_s": load_inputs_elapsed,
        "init_gridworld_wall_time_s": gridworld_init_elapsed,
        "init_wall_time_s": init_elapsed,
        "loop_wall_time_s": loop_elapsed,
        "postprocess_wall_time_s": 0.0,
        "program_wall_time_s": 0.0,
        "steps_executed": steps,
        "step_wall_time_avg_ms": avg_step_ms,
        "step_wall_time_min_ms": min_step_ms,
        "step_wall_time_max_ms": max_step_ms,
        "step_times_csv": str(step_timing_path),
        "timing_mode": "full_run",
    }
    timing_summary["postprocess_wall_time_s"] = (
        time.perf_counter() - postprocess_start
    )
    timing_summary["program_wall_time_s"] = time.perf_counter() - program_start
    write_timing_summary(timing_summary_path, timing_summary)
    if os.environ.get("parity_print_summary") not in (None, "", "0"):
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(timing_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
