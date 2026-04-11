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

    world = GridWorld(
        args.num_worlds,
        start_cell,
        end_cells,
        rewards,
        walls,
        gpu_sim=args.gpu,
        gpu_id=0,
        network_inputs=network_inputs,
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
