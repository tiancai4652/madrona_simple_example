"""Run the merged flow simulator end-to-end with a user-provided topo + workload.

Usage:
  python run_flow_simulator.py --topo <topo.txt> --workload <dir|npu.N.json> \
      [--npu-count N] [--ring-dims X,Y,Z] [--chunks N] [--max-steps N] \
      [--gpu] [--out-csv out.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "madrona_simple_example" / "src"))

from madrona_simple_example import GridWorld, SystemConfig  # noqa: E402
from madrona_simple_example.chakra.conversion import (  # noqa: E402
    folder_to_int_array,
    nodes_to_int_array,
)
from madrona_simple_example.gridworld import (  # noqa: E402
    load_network_inputs_from_files,
)


def parse_ring_dims(value):
    return tuple(int(x) for x in value.split(","))


def load_workload_rows(workload):
    path = Path(workload)
    if path.is_dir():
        return folder_to_int_array(path, max_workers=0, show_progress=True)
    if path.is_file():
        return [nodes_to_int_array(str(path))]
    raise FileNotFoundError(f"workload path does not exist: {workload}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the merged flow simulator with a topo + workload")
    parser.add_argument("--topo", required=True, help="topology file path")
    parser.add_argument("--workload", required=True,
                        help="Chakra workload dir (npu.<id>.json) or single json")
    parser.add_argument("--npu-count", type=int, default=0,
                        help="NPU count (default: derived from workload files)")
    parser.add_argument("--ring-dims", default=None,
                        help="ring dims as X,Y,Z (default: derived from npu count)")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out-csv", default=None,
                        help="optional path to write flow completion CSV")
    args = parser.parse_args()

    rows = load_workload_rows(args.workload)
    npu_count = args.npu_count or len(rows)
    if npu_count != len(rows):
        sys.exit(f"error: --npu-count {npu_count} != workload rows {len(rows)}")
    ring_dims = parse_ring_dims(args.ring_dims) if args.ring_dims else (npu_count, 1, 1)
    config = SystemConfig(ring_dims=ring_dims, chunks_num=args.chunks,
                          npu_count=npu_count)

    network_inputs = load_network_inputs_from_files(args.topo, None)
    walls = np.zeros((1, 1), dtype=bool)
    rewards = np.zeros((1, 1), dtype=np.float32)
    start_cell = np.array([0, 0])
    end_cells = np.array([[0, 0]], dtype=np.int32)

    print(f"[sim] topo={args.topo}")
    print(f"[sim] workload={args.workload} npus={npu_count} ring_dims={ring_dims} "
          f"chunks={args.chunks} gpu={args.gpu}")

    world = GridWorld(
        num_worlds=1,
        start_cell=start_cell,
        end_cells=end_cells,
        rewards=rewards,
        walls=walls,
        gpu_sim=args.gpu,
        network_inputs=network_inputs,
        system_workload=rows,
        system_config=config,
    )

    done_step = 0
    for step in range(1, args.max_steps + 1):
        world.step()
        if step % 10 == 0 or step == 1:
            print(f"[sim] step={step} time_ms={world.simulation_time():.4f} "
                  f"defs={world.num_flow_defs()} "
                  f"pending={world.num_pending_flows()} "
                  f"events={world.num_delayed_events()} "
                  f"tags={world.num_active_tags()} "
                  f"completions={world.num_flow_completions()}", flush=True)
        status = world.system_status()
        if status["failed"]:
            print(f"[sim] FAILED at step {step}: {status}")
            sys.exit(1)
        if status["finished"]:
            done_step = step
            break

    status = world.system_status()
    print(f"[sim] final status: {status}")
    if done_step == 0:
        sys.exit(f"error: system did not finish in {args.max_steps} steps, "
                 f"time={world.simulation_time()}")

    print(f"[sim] finished at step {done_step}, "
          f"sim_time_ms={world.simulation_time()}")
    print(f"[sim] flows completed={world.num_flow_completions()} "
          f"defs={world.num_flow_defs()} "
          f"pending={world.num_pending_flows()} "
          f"events={world.num_delayed_events()} "
          f"tags={world.num_active_tags()}")
    if args.out_csv:
        world.write_flow_completion_csv(args.out_csv)
        print(f"[sim] wrote flow completions to {args.out_csv}")
    print("[sim] RUN OK")


if __name__ == "__main__":
    main()
