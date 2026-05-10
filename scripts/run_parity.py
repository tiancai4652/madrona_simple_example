import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "madrona_simple_example" / "src"
PKG_DIR = SRC_DIR / "madrona_simple_example"
BUILD_DIR = ROOT / "madrona_simple_example" / "build"


def _load_local_package():
    pkg_name = "madrona_simple_example"
    ext_name = f"{pkg_name}._madrona_simple_example_cpp"
    ext_candidates = sorted(BUILD_DIR.glob("_madrona_simple_example_cpp*.so"))
    if not ext_candidates:
        raise ImportError(
            f"missing built extension under {BUILD_DIR}; "
            "run `cmake --build madrona_simple_example/build -j4` first"
        )

    for mod_name in list(sys.modules.keys()):
        if mod_name == pkg_name or mod_name.startswith(pkg_name + "."):
            del sys.modules[mod_name]

    ext_spec = importlib.util.spec_from_file_location(ext_name, ext_candidates[0])
    if ext_spec is None or ext_spec.loader is None:
        raise ImportError(f"failed to load extension spec from {ext_candidates[0]}")
    ext_mod = importlib.util.module_from_spec(ext_spec)
    sys.modules[ext_name] = ext_mod
    ext_spec.loader.exec_module(ext_mod)

    init_py = PKG_DIR / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_py,
        submodule_search_locations=[str(PKG_DIR)],
    )
    if pkg_spec is None or pkg_spec.loader is None:
        raise ImportError(f"failed to load package spec from {init_py}")
    pkg_mod = importlib.util.module_from_spec(pkg_spec)
    sys.modules[pkg_name] = pkg_mod
    pkg_spec.loader.exec_module(pkg_mod)
    return pkg_mod


_pkg = _load_local_package()
GridWorld = _pkg.GridWorld
load_network_inputs_from_files = _pkg.load_network_inputs_from_files


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


def main():
    args = parse_args()
    t_start = time.time()
    _pstage("main() entered", t_start)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _pstage(f"loading inputs: topo={args.topo} flows={args.flows}", t_start)
    network_inputs = load_network_inputs_from_files(Path(args.topo), Path(args.flows))
    walls, rewards, end_cells, start_cell = build_grid_inputs()
    _pstage("inputs loaded", t_start)

    prior_weights = []
    if args.prior_weights:
        prior_weights = [float(x) for x in args.prior_weights.split(',')]

    _pstage(
        f"constructing GridWorld(gpu={bool(args.gpu)}, pfc={bool(args.pfc)}, "
        f"num_worlds={args.num_worlds}) ...",
        t_start,
    )
    world = GridWorld(
        args.num_worlds,
        start_cell,
        end_cells,
        rewards,
        walls,
        gpu_sim=bool(args.gpu),
        gpu_id=0,
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
    _pstage("GridWorld constructed (GPU init + launch graph built)", t_start)

    if os.environ.get("init_log_print_enabled") not in (None, "", "0"):
        return

    log_capture_mode = _is_log_capture_mode()
    progress_every = int(os.environ.get("PARITY_PROGRESS_EVERY", "500"))

    steps = 0
    t0 = time.time()
    t_prev = t0
    steps_prev = 0
    step_phase_rows = []
    if not log_capture_mode:
        print(
            f"[parity] start loop: max_steps={args.max_steps} "
            f"gpu={bool(args.gpu)} pfc={bool(args.pfc)} "
            f"progress_every={progress_every}",
            flush=True,
        )

    while steps < args.max_steps and not should_stop(world):
        world.step()
        steps += 1
        phase_times = world.last_step_phase_times()
        if phase_times.get("step", 0) == 0:
            phase_times["step"] = steps
        step_phase_rows.append(phase_times)

        if (
            not log_capture_mode
            and progress_every > 0
            and steps % progress_every == 0
        ):
            now = time.time()
            dt = max(now - t_prev, 1e-6)
            sps = (steps - steps_prev) / dt
            elapsed = now - t0
            print(
                f"[parity] step={steps}/{args.max_steps} "
                f"sim_t={world.simulation_time():.3f} "
                f"pend={world.num_pending_flows()} "
                f"delayed={world.num_delayed_events()} "
                f"active={world.num_active_tags()} "
                f"sps={sps:.1f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            t_prev = now
            steps_prev = steps

    if not log_capture_mode:
        print(
            f"[parity] loop done: steps={steps} "
            f"elapsed={time.time() - t0:.1f}s "
            f"stopped_cleanly={should_stop(world)}",
            flush=True,
        )

    completion_path = out_dir / "flow_completion_times.csv"
    world.write_flow_completion_csv(completion_path)
    phase_times_path = out_dir / "madrona_step_phase_times.csv"
    if step_phase_rows:
        fieldnames = [
            "step",
            "total_wall_time_s",
            "schedule_wall_time_s",
            "deliver_wall_time_s",
            "ingress_wall_time_s",
            "alloc_wall_time_s",
            "pfc_emit_wall_time_s",
            "clear_dt_wall_time_s",
            "buffer_progress_wall_time_s",
        ]
        with open(phase_times_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in step_phase_rows:
                writer.writerow({k: row.get(k, 0.0) for k in fieldnames})

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
        "step_phase_times_csv": str(phase_times_path),
        "stopped_cleanly": should_stop(world),
    }

    summary_path = out_dir / "madrona_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if os.environ.get("parity_print_summary") not in (None, "", "0"):
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
