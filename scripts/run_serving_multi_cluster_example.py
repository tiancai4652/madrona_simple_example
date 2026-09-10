"""CPU example with four P workers, four D workers, and 20 requests."""

import argparse
import collections
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
    # A non-zero, pair-specific comm_para gives SEND/RECV a persistent key.
    comm_para = (src + 1) * 100 + node_id
    return Node(
        id=str(node_id),
        name=f"{node_type.lower()}-{src}-{dst}-{node_id}",
        type=node_type,
        dataDeps=[str(dep) for dep in deps],
        durationMicros=0,
        attr=[
            Attribute("comm_para", uint32Val=comm_para),
            Attribute("comm_size", uint64Val=size_bytes),
            Attribute("comm_src", uint64Val=src),
            Attribute("comm_dst", uint64Val=dst),
        ],
    )


def make_rank_dag(src, dst, receive):
    """Build a 12-node DAG with two communication synchronization points."""
    comm_type = "COMM_RECV_NODE" if receive else "COMM_SEND_NODE"
    return encode_nodes([
        comp(0, [], 2),
        comp(1, [0], 3),
        comp(2, [0], 2),
        communication(comm_type, 3, [1], src, dst, 512),
        comp(4, [3], 2),
        comp(5, [2], 4),
        communication(comm_type, 6, [4], src, dst, 1024),
        comp(7, [5], 2),
        comp(8, [6], 3),
        comp(9, [7, 8], 2),
        comp(10, [9], 2),
        comp(11, [10], 1),
    ])


def make_workload():
    """Create one sender/receiver DAG pair for every logical worker."""
    rows = []
    for src in range(0, 16, 2):
        dst = src + 1
        rows.append(make_rank_dag(src, dst, receive=False))
        rows.append(make_rank_dag(src, dst, receive=True))
    return rows


def make_network():
    """Two leaf switches and one spine separate P ranks from D ranks."""
    host_ids = np.arange(16, dtype=np.int32)
    host_leaf = np.array([16] * 8 + [17] * 8, dtype=np.int32)
    return {
        "node_ids": np.arange(19, dtype=np.int32),
        "node_types": np.array([0] * 16 + [1, 1, 1], dtype=np.int32),
        "node_port_bws": np.array(
            [1e8] * 16 + [4e8, 4e8, 8e8], dtype=np.float64),
        "link_srcs": np.concatenate((
            host_ids, np.array([16, 17], dtype=np.int32))),
        "link_dsts": np.concatenate((
            host_leaf, np.array([18, 18], dtype=np.int32))),
        "link_delays": np.concatenate((
            np.full(16, 0.001, dtype=np.float64),
            np.full(2, 0.002, dtype=np.float64))),
        "link_bandwidths": np.concatenate((
            np.full(16, 1e8, dtype=np.float64),
            np.full(2, 4e8, dtype=np.float64))),
        "flow_ids": np.array([], dtype=np.int64),
        "flow_src_nodes": np.array([], dtype=np.int32),
        "flow_dst_nodes": np.array([], dtype=np.int32),
        "flow_sizes": np.array([], dtype=np.float64),
        "flow_start_times": np.array([], dtype=np.float64),
        "flow_priorities": np.array([], dtype=np.int32),
    }


PROMPT_LENGTHS = [
    4, 8, 6, 2, 10, 5, 7, 3, 12, 9,
    4, 11, 6, 2, 8, 5, 13, 7, 3, 10,
]
OUTPUT_LENGTHS = [
    3, 1, 5, 2, 4, 2, 6, 3, 1, 5,
    4, 2, 3, 7, 2, 4, 1, 5, 3, 6,
]


def make_requests():
    requests = []
    for idx, (prompt, output) in enumerate(
            zip(PROMPT_LENGTHS, OUTPUT_LENGTHS)):
        if idx < 8:
            arrival_ns = 0
        elif idx < 14:
            arrival_ns = 5_000
        else:
            arrival_ns = 12_000
        requests.append({
            "request_id": 200 + idx,
            "arrival_time_ns": arrival_ns,
            "prompt_len": prompt,
            "output_len": output,
        })
    return requests


WORKLOAD_PARAMS_TABLE = [
    {
        "stage": "prefill",
        "node_type": "COMP_NODE",
        "batch_size": -1,
        "sequence_tokens": -1,
        "duration_ns": 2_000,
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
]

STATE_NAMES = {
    0: "Empty",
    1: "WaitingArrival",
    2: "WaitingPrefill",
    3: "Prefilling",
    4: "KvTransferring",
    5: "WaitingDecode",
    6: "Decoding",
    7: "Finished",
    8: "Rejected",
}

FRAME_PHASES = (
    "ServingPre(arrival/P schedule)",
    "NPUFlowCheck",
    "RecvCheck",
    "KVCompletion",
    "ChakraRemove",
    "ChakraProcess",
    "ServingPost(P/D completion)",
    "FlowCreate/Schedule",
    "NetworkIngress",
    "Bandwidth/PFC/Emit",
    "BufferProgress",
    "Finish/Status",
)


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
            ring_dims=(16, 1, 1),
            chunks_num=1,
            npu_count=16,
        ),
        inference_config=InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1), (2, 3), (4, 5), (6, 7)),
            decode_workers=((8, 9), (10, 11), (12, 13), (14, 15)),
            p_max_batch=3,
            p_max_tokens=18,
            d_max_batch=4,
            d_max_tokens=80,
            num_layers=4,
            num_kv_heads=4,
            head_dim=8,
            bytes_per_elem=2,
            kv_partition_factor=2,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        ),
        request_trace=make_requests(),
        workload_params_table=WORKLOAD_PARAMS_TABLE,
    )


def format_counts(stats, worker_key):
    counts = collections.Counter(row[worker_key] for row in stats)
    return ", ".join(
        f"{worker_key[0].upper()}{worker}={counts[worker]}"
        for worker in sorted(counts))


def _request_events(previous, current):
    request_id = current["request_id"]
    prefix = f"request={request_id}"
    if previous is None:
        events = [
            f"{prefix} initialized state={STATE_NAMES[current['state']]}"
        ]
        if current["p_worker"] >= 0:
            events.append(
                f"{prefix} P-route=P{current['p_worker']} "
                f"p_start_ns={current['p_start_ns']}")
        return events

    events = []
    old_state = previous["state"]
    new_state = current["state"]
    if old_state != new_state:
        events.append(
            f"{prefix} state={STATE_NAMES[old_state]}"
            f"->{STATE_NAMES[new_state]}")

    if previous["p_worker"] != current["p_worker"]:
        events.append(
            f"{prefix} P-route=P{current['p_worker']} "
            f"prompt_tokens={current['prompt_len']}")
    if previous["p_start_ns"] != current["p_start_ns"]:
        events.append(
            f"{prefix} Prefill-start ns={current['p_start_ns']}")
    if previous["p_finish_ns"] != current["p_finish_ns"]:
        events.append(
            f"{prefix} Prefill-finish ns={current['p_finish_ns']}")

    if previous["d_worker"] != current["d_worker"]:
        events.append(
            f"{prefix} D-route=D{current['d_worker']} "
            f"route_ns={current['d_route_ns']}")
    if previous["kv_start_ns"] != current["kv_start_ns"]:
        events.append(
            f"{prefix} KV-start ns={current['kv_start_ns']} "
            f"bytes={current['kv_bytes']} "
            f"shards={current['kv_shards']}")
    if previous["kv_shards_done"] != current["kv_shards_done"]:
        events.append(
            f"{prefix} KV-progress "
            f"{previous['kv_shards_done']}"
            f"->{current['kv_shards_done']}"
            f"/{current['kv_shards']}")
    if previous["kv_done_ns"] != current["kv_done_ns"]:
        events.append(
            f"{prefix} KV-finish ns={current['kv_done_ns']}")

    if previous["d_start_ns"] != current["d_start_ns"]:
        events.append(
            f"{prefix} Decode-admit D{current['d_worker']} "
            f"ns={current['d_start_ns']}")
    if previous["output_done"] != current["output_done"]:
        events.append(
            f"{prefix} Decode-token "
            f"{previous['output_done']}->{current['output_done']}"
            f"/{current['output_len']}")
    if previous["first_token_ns"] != current["first_token_ns"]:
        events.append(
            f"{prefix} first-token ns={current['first_token_ns']}")
    if previous["finish_ns"] != current["finish_ns"]:
        events.append(
            f"{prefix} request-finish ns={current['finish_ns']} "
            f"e2e_ns={current['e2e_ns']}")
    return events


def _write_frame_log(
        stream, frame, simulation_time_ms, status, previous, current):
    stream.write(
        f"FRAME {frame:04d} simulation_time_ms={simulation_time_ms:.9f} "
        f"finished={int(status['finished'])} "
        f"failed={int(status['failed'])}\n")
    stream.write("  task_graph: " + " -> ".join(FRAME_PHASES) + "\n")

    previous_by_id = (
        {} if previous is None
        else {row["request_id"]: row for row in previous}
    )
    events = []
    for row in current:
        events.extend(_request_events(
            previous_by_id.get(row["request_id"]), row))
    if events:
        stream.write("  business_events:\n")
        for event in events:
            stream.write(f"    - {event}\n")
    else:
        stream.write(
            "  business_events: none; DAG/network discrete-event "
            "state progressed or waited for the next event\n")
    stream.write("\n")


def run(trace_file=None):
    world = make_world()
    trace_stream = None
    previous_stats = None
    try:
        if trace_file is not None:
            trace_file = Path(trace_file)
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_stream = trace_file.open("w", encoding="utf-8")
            trace_stream.write(
                "# Serving multi-cluster frame log\n"
                "# One FRAME equals one world.step() task-graph execution.\n"
                "# simulation_time_ms is simulated time, not host wall time.\n\n"
            )

        for step in range(1, 10_001):
            world.step()
            status = world.system_status()
            current_stats = world.inference_stats()
            if trace_stream is not None:
                _write_frame_log(
                    trace_stream,
                    step,
                    world.simulation_time(),
                    status,
                    previous_stats,
                    current_stats,
                )
            previous_stats = current_stats
            if status["failed"]:
                raise RuntimeError(
                    f"simulation failed at step {step}: {status}")
            if status["finished"]:
                break
        else:
            raise RuntimeError(
                f"simulation did not finish: {world.system_status()}")
    finally:
        if trace_stream is not None:
            trace_stream.close()

    stats = world.inference_stats()
    print(
        f"finished in {step} steps, "
        f"simulation_time_ms={world.simulation_time():.9f}")
    print("P routing:", format_counts(stats, "p_worker"))
    print("D routing:", format_counts(stats, "d_worker"))
    print(
        "request arrival prompt output P D p_start p_finish "
        "kv_done first_token finish ttft tpot")
    for row in stats:
        print(
            f"{row['request_id']:>7} "
            f"{row['arrival_time_ns']:>7} "
            f"{row['prompt_len']:>6} "
            f"{row['output_len']:>6} "
            f"{row['p_worker']:>1} "
            f"{row['d_worker']:>1} "
            f"{row['p_start_ns']:>7} "
            f"{row['p_finish_ns']:>8} "
            f"{row['kv_done_ns']:>7} "
            f"{row['first_token_ns']:>11} "
            f"{row['finish_ns']:>6} "
            f"{row['ttft_ns']:>5} "
            f"{row['tpot_ns']:>6.1f}")

    if len(stats) != 20:
        raise AssertionError(f"expected 20 requests, got {len(stats)}")
    if any(row["state"] != 7 for row in stats):
        raise AssertionError("not every request reached Finished")
    if len({row["p_worker"] for row in stats}) != 4:
        raise AssertionError("P queue-depth routing did not use all workers")
    if len({row["d_worker"] for row in stats}) != 4:
        raise AssertionError("D queue-depth routing did not use all workers")
    if any(row["kv_shards_done"] != 2 for row in stats):
        raise AssertionError("not every request completed both KV shards")
    if trace_file is not None:
        print(f"frame trace: {trace_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 16-rank, 20-request PD routing example.")
    parser.add_argument(
        "--trace-file",
        type=Path,
        help="write one detailed record for every simulation frame",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(trace_file=args.trace_file)
