import argparse
import json
import os
import sys
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


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    network_inputs = load_network_inputs_from_files(Path(args.topo), Path(args.flows))
    walls, rewards, end_cells, start_cell = build_grid_inputs()

    prior_weights = []
    if args.prior_weights:
        prior_weights = [float(x) for x in args.prior_weights.split(',')]

    world = GridWorld(
        args.num_worlds,
        start_cell,
        end_cells,
        rewards,
        walls,
        gpu_sim=False,
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

    if os.environ.get("init_log_print_enabled") not in (None, "", "0"):
        return

    steps = 0
    while steps < args.max_steps and not should_stop(world):
        world.step()
        steps += 1

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
    if os.environ.get("parity_print_summary") not in (None, "", "0"):
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
