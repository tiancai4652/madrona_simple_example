"""Run the PD serving simulator with Chakra workloads and a request trace."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "madrona_simple_example" / "src"))

from madrona_simple_example import (  # noqa: E402
    GridWorld,
    InferenceConfig,
    SystemConfig,
)
from madrona_simple_example.chakra.conversion import (  # noqa: E402
    folder_to_int_array,
    nodes_to_int_array,
)
from madrona_simple_example.gridworld import (  # noqa: E402
    load_network_inputs_from_files,
)


def parse_ring_dims(value):
    return tuple(int(part) for part in value.split(","))


def parse_workers(value):
    workers = []
    for item in value.split(","):
        bounds = item.strip().split("-")
        if len(bounds) == 1:
            start = end = int(bounds[0])
        elif len(bounds) == 2:
            start, end = (int(bound) for bound in bounds)
        else:
            raise argparse.ArgumentTypeError(
                f"invalid worker range: {item!r}")
        if start < 0 or end < start:
            raise argparse.ArgumentTypeError(
                f"invalid worker range: {item!r}")
        workers.append(tuple(range(start, end + 1)))
    if not workers:
        raise argparse.ArgumentTypeError("at least one worker is required")
    return tuple(workers)


def load_workload_rows(workload):
    path = Path(workload)
    if path.is_dir():
        return folder_to_int_array(path, max_workers=0, show_progress=True)
    if path.is_file():
        return [nodes_to_int_array(str(path))]
    raise FileNotFoundError(f"workload path does not exist: {workload}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the PD serving simulator")
    parser.add_argument("--topo", required=True, help="topology file path")
    parser.add_argument("--workload", required=True,
                        help="Chakra workload dir or single npu JSON")
    parser.add_argument("--request-trace", required=True,
                        help="request trace JSON, JSONL, or CSV")
    parser.add_argument("--workload-params", default=None,
                        help="optional workload params table JSON, JSONL, or CSV")
    parser.add_argument("--npu-count", type=int, default=0)
    parser.add_argument("--ring-dims", default=None,
                        help="ring dimensions as X,Y,Z")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--prefill-workers", required=True,
                        type=parse_workers, help="worker rank ranges, for example 0-3,4-7 (widths may differ per worker)")
    parser.add_argument("--decode-workers", required=True,
                        type=parse_workers, help="worker rank ranges, for example 8-11,12-15 (widths may differ per worker)")
    parser.add_argument("--p-max-batch", type=int, default=16)
    parser.add_argument("--p-max-tokens", type=int, default=8192)
    parser.add_argument("--d-max-batch", type=int, default=16)
    parser.add_argument("--d-max-tokens", type=int, default=32768)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bytes-per-elem", type=int, default=2)
    parser.add_argument("--kv-partition-factor", type=int, default=0,
                        help="0 = auto (GQA: each P worker rank streams the "
                             "KV of the head block it holds to the D rank "
                             "owning those heads; replicated ranks stay "
                             "silent); a positive value must restate the "
                             "same GQA layout (== every prefill worker's "
                             "rank count)")
    parser.add_argument("--prefill-reference-tokens", type=int, default=1)
    parser.add_argument("--decode-reference-tokens", type=int, default=1)
    parser.add_argument("--p-max-wait-ns", type=int, default=0)
    parser.add_argument("--d-max-wait-ns", type=int, default=0,
                        help="D-side batch-assembly timeout in ns; 0 keeps "
                             "the plain continuous batching behavior (start "
                             "as soon as idle)")
    parser.add_argument("--kv-mode", choices=("gqa", "mla"), default="gqa",
                        help="KV cache layout: gqa (default, per-head-block "
                             "sharding across P/D ranks, unchanged legacy "
                             "behavior) or mla (Multi-head Latent Attention, "
                             "e.g. DeepSeek-V3: one full latent stream P0->"
                             "each D rank, shards == d_width, point-to-point "
                             "over the scaleout network, no divisibility "
                             "requirements)")
    parser.add_argument("--kv-latent-dim", type=int, default=0,
                        help="MLA latent width per layer/token = "
                             "kv_lora_rank + qk_rope_head_dim (DeepSeek-V3: "
                             "576); required (>0) with --kv-mode mla, must "
                             "stay 0 with --kv-mode gqa")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out-csv", default=None,
                        help="optional serving statistics CSV path")
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--print-every", type=int, default=10)
    return parser


def main():
    args = build_parser().parse_args()
    rows = load_workload_rows(args.workload)
    npu_count = args.npu_count or len(rows)
    if npu_count != len(rows):
        sys.exit(
            f"error: --npu-count {npu_count} != workload rows {len(rows)}")
    ring_dims = (
        parse_ring_dims(args.ring_dims)
        if args.ring_dims else (npu_count, 1, 1)
    )
    system_config = SystemConfig(
        ring_dims=ring_dims, chunks_num=args.chunks, npu_count=npu_count)
    inference_config = InferenceConfig(
        enabled=True,
        prefill_workers=args.prefill_workers,
        decode_workers=args.decode_workers,
        p_max_batch=args.p_max_batch,
        p_max_tokens=args.p_max_tokens,
        d_max_batch=args.d_max_batch,
        d_max_tokens=args.d_max_tokens,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        bytes_per_elem=args.bytes_per_elem,
        kv_partition_factor=args.kv_partition_factor,
        prefill_reference_tokens=args.prefill_reference_tokens,
        decode_reference_tokens=args.decode_reference_tokens,
        p_max_wait_ns=args.p_max_wait_ns,
        d_max_wait_ns=args.d_max_wait_ns,
        kv_mode={"gqa": 0, "mla": 1}[args.kv_mode],
        kv_latent_dim=args.kv_latent_dim,
    )

    network_inputs = load_network_inputs_from_files(args.topo, None)
    world = GridWorld(
        num_worlds=1,
        start_cell=np.array([0, 0]),
        end_cells=np.array([[0, 0]], dtype=np.int32),
        rewards=np.zeros((1, 1), dtype=np.float32),
        walls=np.zeros((1, 1), dtype=bool),
        gpu_sim=args.gpu,
        network_inputs=network_inputs,
        system_workload=rows,
        system_config=system_config,
        inference_config=inference_config,
        request_trace=args.request_trace,
        workload_params_table=args.workload_params_table,
    )

    finished_step = 0
    for step in range(1, args.max_steps + 1):
        world.step()
        status = world.system_status()
        if step == 1 or step % args.print_every == 0:
            print(
                f"[serving] step={step} "
                f"time_ms={world.simulation_time():.4f} status={status}",
                flush=True)
        if status["failed"]:
            sys.exit(
                f"error: inference simulation failed at step {step}, "
                f"error_code={status['error_code']}")
        if status["finished"]:
            finished_step = step
            break

    if not finished_step:
        sys.exit(
            f"error: inference simulation did not finish in "
            f"{args.max_steps} steps")
    if args.out_csv:
        world.write_inference_stats_csv(args.out_csv)
        print(f"[serving] wrote statistics to {args.out_csv}")
    print(
        f"[serving] finished at step {finished_step}, "
        f"sim_time_ms={world.simulation_time():.4f}")


if __name__ == "__main__":
    main()
