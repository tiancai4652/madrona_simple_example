"""CPU verification for the Stage-B Chakra system integration."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "madrona_simple_example" / "src"))

from madrona_simple_example import GridWorld, SystemConfig  # noqa: E402
from madrona_simple_example.chakra.conversion import (  # noqa: E402
    node_to_int_array,
)
from madrona_simple_example.chakra.parser import Attribute, Node  # noqa: E402
from madrona_simple_example.gridworld import (  # noqa: E402
    load_network_inputs_from_files,
)


TOPO = ROOT / "jiuding_dodsim/examples/leafspine128/leafspine_h128_topo.txt"


def make_world(workload=None, config=None):
    return GridWorld(
        num_worlds=1,
        start_cell=(0, 0),
        end_cells=np.array([[0, 0]], dtype=np.int32),
        rewards=np.zeros((1, 1), dtype=np.float32),
        walls=np.zeros((1, 1), dtype=bool),
        network_inputs=load_network_inputs_from_files(TOPO, None),
        system_workload=workload,
        system_config=config,
    )


def run_until_done(world, limit=100):
    for step in range(1, limit + 1):
        world.step()
        if world.system_finished() or world.system_failed():
            return step
    raise AssertionError(
        f"system did not finish in {limit} steps: "
        f"status={world.system_status()}, time={world.simulation_time()}, "
        f"flows={world.num_flow_defs()}, events={world.num_delayed_events()}")


def encoded_node(node_type, node_id, attrs=None, deps=None, duration=0):
    return node_to_int_array(Node(
        name=f"node-{node_id}",
        type=node_type,
        durationMicros=duration,
        attr=[] if attrs is None else attrs,
        id=str(node_id),
        dataDeps=[] if deps is None else [str(dep) for dep in deps],
    ))


def point_to_point_attrs(src, dst, size):
    return [
        Attribute("comm_size", uint64Val=size),
        Attribute("comm_src", uint64Val=src),
        Attribute("comm_dst", uint64Val=dst),
    ]


network_only = make_world()
network_only.step()
assert network_only.num_flow_defs() == 0
assert not network_only.system_status()["initialized"]
print("[PASS] B1 network-only mode remains available")


comp_world = make_world(
    [[*encoded_node("COMP_NODE", 1, duration=5)]],
    SystemConfig(ring_dims=(1, 1, 1), chunks_num=1, npu_count=1),
)
comp_steps = run_until_done(comp_world, 10)
assert not comp_world.system_failed()
assert abs(comp_world.simulation_time() - 0.005) < 1e-12
print(f"[PASS] B2 computation event completed at 5 us in {comp_steps} steps")


attrs = point_to_point_attrs(0, 1, 1024 * 1024)
p2p_world = make_world(
    [
        encoded_node("COMM_SEND_NODE", 1, attrs=attrs),
        encoded_node("COMM_RECV_NODE", 1, attrs=attrs),
    ],
    SystemConfig(ring_dims=(2, 1, 1), chunks_num=1, npu_count=2),
)
p2p_steps = run_until_done(p2p_world, 30)
records = p2p_world.flow_completions()
assert not p2p_world.system_failed()
assert len(records) == 1
assert records[0]["src_node"] == 0 and records[0]["dst_node"] == 1
assert records[0]["size_bytes"] == 1024 * 1024
assert p2p_world.num_flow_defs() == 0
assert p2p_world.num_active_tags() == 0
print(f"[PASS] B3 SEND/RECV network round trip completed in {p2p_steps} steps")


try:
    make_world(
        [[*encoded_node("COMP_NODE", 1, duration=1)]],
        SystemConfig(ring_dims=(2, 1, 1), chunks_num=1, npu_count=2),
    )
except ValueError as exc:
    assert "workload has" in str(exc)
else:
    raise AssertionError("invalid workload row count was accepted")
print("[PASS] B4 Python rejects inconsistent system input")

print("all Stage-B CPU integration cases passed")
