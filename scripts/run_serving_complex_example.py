"""Self-contained CPU example for a non-trivial PD serving scenario."""

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
    node_to_int_array,
)
from madrona_simple_example.chakra.parser import Attribute, Node  # noqa: E402


def encode_nodes(nodes):
    encoded = []
    for node in nodes:
        encoded.extend(node_to_int_array(node))
    return encoded


def comp(node_id, deps, duration_us):
    return Node(
        id=str(node_id),
        name=f"compute-{node_id}",
        type="COMP_NODE",
        dataDeps=[str(dep) for dep in deps],
        durationMicros=duration_us,
        attr=[],
    )


def communication(node_type, node_id, deps, src, dst, size_bytes):
    return Node(
        id=str(node_id),
        name=f"{node_type.lower()}-{src}-{dst}",
        type=node_type,
        dataDeps=[str(dep) for dep in deps],
        durationMicros=0,
        attr=[
            Attribute("comm_para", uint32Val=0),
            Attribute("comm_size", uint64Val=size_bytes),
            Attribute("comm_src", uint64Val=src),
            Attribute("comm_dst", uint64Val=dst),
        ],
    )


def make_pair_workloads(src, dst):
    """Create compute -> SEND/RECV -> compute DAGs for a two-rank worker."""
    sender = encode_nodes([
        comp(0, [], 2),
        communication("COMM_SEND_NODE", 1, [0], src, dst, 256),
        comp(2, [1], 1),
    ])
    receiver = encode_nodes([
        comp(0, [], 2),
        communication("COMM_RECV_NODE", 1, [0], src, dst, 256),
        comp(2, [1], 1),
    ])
    return sender, receiver


def make_workload():
    """Build four two-rank workers: P0, P1, D0, D1."""
    rows = []
    for src, dst in ((0, 1), (2, 3), (4, 5), (6, 7)):
        sender, receiver = make_pair_workloads(src, dst)
        rows.extend((sender, receiver))
    return rows


def make_network():
    """Build eight hosts connected through one switch."""
    host_ids = np.arange(8, dtype=np.int32)
    return {
        "node_ids": np.arange(9, dtype=np.int32),
        "node_types": np.array([0] * 8 + [1], dtype=np.int32),
        "node_port_bws": np.full(9, 1e8, dtype=np.float64),
        "link_srcs": host_ids,
        "link_dsts": np.full(8, 8, dtype=np.int32),
        "link_delays": np.full(8, 0.001, dtype=np.float64),
        "link_bandwidths": np.full(8, 1e8, dtype=np.float64),
        "flow_ids": np.array([], dtype=np.int64),
        "flow_src_nodes": np.array([], dtype=np.int32),
        "flow_dst_nodes": np.array([], dtype=np.int32),
        "flow_sizes": np.array([], dtype=np.float64),
        "flow_start_times": np.array([], dtype=np.float64),
        "flow_priorities": np.array([], dtype=np.int32),
    }


REQUESTS = [
    {
        "request_id": 100,
        "arrival_time_ns": 0,
        "prompt_len": 4,
        "output_len": 3,
    },
    {
        "request_id": 101,
        "arrival_time_ns": 0,
        "prompt_len": 3,
        "output_len": 1,
    },
    {
        "request_id": 102,
        "arrival_time_ns": 0,
        "prompt_len": 2,
        "output_len": 2,
    },
    {
        "request_id": 103,
        "arrival_time_ns": 0,
        "prompt_len": 1,
        "output_len": 4,
    },
    {
        "request_id": 104,
        "arrival_time_ns": 2_000,
        "prompt_len": 5,
        "output_len": 2,
    },
]


WORKLOAD_PARAMS_TABLE = [
    {
        "stage": "prefill",
        "node_type": "COMP_NODE",
        "batch_size": -1,
        "sequence_tokens": -1,
        "duration_ns": 3_000,
        "comm_size_bytes": 0,
    },
    {
        "stage": "decode",
        "node_type": "COMP_NODE",
        "batch_size": -1,
        "sequence_tokens": -1,
        "duration_ns": 1_000,
        "comm_size_bytes": 0,
    },
    {
        "stage": "prefill",
        "node_type": "COMM_SEND_NODE",
        "batch_size": -1,
        "sequence_tokens": -1,
        "duration_ns": 0,
        "comm_size_bytes": 256,
    },
    {
        "stage": "decode",
        "node_type": "COMM_SEND_NODE",
        "batch_size": -1,
        "sequence_tokens": -1,
        "duration_ns": 0,
        "comm_size_bytes": 256,
    },
]


def make_world():
    return GridWorld(
        num_worlds=1,
        start_cell=(0, 0),
        end_cells=np.array([[0, 0]], dtype=np.int32),
        rewards=np.zeros((1, 1), dtype=np.float32),
        walls=np.zeros((1, 1), dtype=bool),
        network_inputs=make_network(),
        system_workload=make_workload(),
        system_config=SystemConfig(
            ring_dims=(8, 1, 1),
            chunks_num=1,
            npu_count=8,
        ),
        inference_config=InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1), (2, 3)),
            decode_workers=((4, 5), (6, 7)),
            p_max_batch=2,
            p_max_tokens=6,
            d_max_batch=3,
            d_max_tokens=20,
            num_layers=2,
            num_kv_heads=2,
            head_dim=4,
            bytes_per_elem=2,
            kv_partition_factor=2,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        ),
        request_trace=REQUESTS,
        workload_params_table=WORKLOAD_PARAMS_TABLE,
    )


def run():
    world = make_world()
    for step in range(1, 2_001):
        world.step()
        status = world.system_status()
        if status["failed"]:
            raise RuntimeError(
                f"simulation failed at step {step}: {status}")
        if status["finished"]:
            break
    else:
        raise RuntimeError(
            f"simulation did not finish: {world.system_status()}")

    print(
        f"finished in {step} steps, "
        f"simulation_time_ms={world.simulation_time():.9f}")
    print(
        "request  P  D  prompt  output  kv_bytes  "
        "p_start  p_finish  kv_done  first_token  finish")
    for row in world.inference_stats():
        print(
            f"{row['request_id']:>7}  "
            f"{row['p_worker']:>1}  "
            f"{row['d_worker']:>1}  "
            f"{row['prompt_len']:>6}  "
            f"{row['output_done']:>6}  "
            f"{row['kv_bytes']:>8}  "
            f"{row['p_start_ns']:>7}  "
            f"{row['p_finish_ns']:>8}  "
            f"{row['kv_done_ns']:>7}  "
            f"{row['first_token_ns']:>11}  "
            f"{row['finish_ns']:>6}")


if __name__ == "__main__":
    run()
