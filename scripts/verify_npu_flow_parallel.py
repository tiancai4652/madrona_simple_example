"""Stage-A2 per-NPU flow bookkeeping verification (CPU).

Covers the cases described in
report/2-merge-plan/实施/阶段A2-per-NPU流并行化验证方案.md that are runnable
against the default build (NUM_FAKE_NPUS = 4 in sim.cpp). The scaling and
pool-recycling stress cases need that constant / FakeSystemDriver::max_rounds
raised and a rebuild, so they are driven from the doc instead.

Usage:
    PYTHONPATH=src python3 scripts/verify_npu_flow_parallel.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "madrona_simple_example" / "src"))

from madrona_simple_example.gridworld import (  # noqa: E402
    GridWorld,
    load_network_inputs_from_files,
)

TOPO = ROOT / "jiuding_dodsim/examples/leafspine128/leafspine_h128_topo.txt"
FLOWS = ROOT / "jiuding_dodsim/examples/leafspine128/leafspine_h128_d8_alltoall_2mb.txt"

# Must match Sim::Sim's NUM_FAKE_NPUS and FakeSystemDriver::max_rounds.
NUM_NPUS = 4
ROUNDS = 3
EXPECTED_DYNAMIC = NUM_NPUS * ROUNDS

failures = []


def check(case, cond, detail):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {case}: {detail}", flush=True)
    if not cond:
        failures.append(case)


def make_world(flow_path):
    ni = load_network_inputs_from_files(TOPO, flow_path)
    return GridWorld(
        num_worlds=1,
        start_cell=(0, 0),
        end_cells=np.array([[0, 0]], dtype=np.int32),
        rewards=np.zeros((1, 1), dtype=np.float32),
        walls=np.zeros((1, 1), dtype=bool),
        network_inputs=ni,
    )


def run(world, max_steps, done):
    peak_defs = 0
    for step in range(max_steps):
        world.step()
        peak_defs = max(peak_defs, world.num_flow_defs())
        if done(world):
            break
    return step + 1, peak_defs


def dynamic_records(world):
    return [r for r in world.flow_completions()
            if 900000 <= r["flow_id"] < 900000 + NUM_NPUS * 1000]


# V1 concurrency: every round must have all NUM_NPUS flows in flight at once,
# which is exactly what the per-NPU bookkeeping buys over the old global queue.
world = make_world(None)
steps, peak_defs = run(world, 400,
                       lambda w: w.num_flow_completions() >= EXPECTED_DYNAMIC)
check("V1 per-NPU concurrency",
      peak_defs == NUM_NPUS,
      f"peak concurrent flow defs = {peak_defs} (expected {NUM_NPUS}), "
      f"converged in {steps} steps")

# V2 completion accounting: counter, exported records and per-NPU flow_id
# segmentation must all agree, and every NPU must finish all its rounds.
recs = dynamic_records(world)
ids = sorted(r["flow_id"] for r in recs)
per_npu = [sum(1 for i in ids if i // 1000 == 900 + n) for n in range(NUM_NPUS)]
check("V2 completion accounting",
      world.num_flow_completions() == EXPECTED_DYNAMIC and
      len(recs) == EXPECTED_DYNAMIC and
      len(set(ids)) == EXPECTED_DYNAMIC and
      all(c == ROUNDS for c in per_npu),
      f"counter={world.num_flow_completions()} records={len(recs)} "
      f"unique_ids={len(set(ids))} per_npu={per_npu}")

# V3 record durability: pool slots are recycled, so records only survive if
# DynamicFlowCompletionLog is doing its job (before it, these were all -1).
fcts = [r["fct_ms"] for r in recs]
check("V3 record durability after pool reuse",
      all(r["flow_id"] >= 0 and r["src_node"] >= 0 and r["size_bytes"] > 0
          for r in recs) and min(fcts) > 0,
      f"all {len(recs)} records populated, fct_ms min/max = "
      f"{min(fcts):.6f}/{max(fcts):.6f}")

# V4 quiescence: nothing may be left dangling once every NPU is Finished --
# in particular numFlowDefs must return to 0, proving each FlowMeta entity
# went back to its NpuFlowPool instead of leaking.
for _ in range(5):
    world.step()
check("V4 quiescence after all rounds",
      world.num_flow_defs() == 0 and world.num_pending_flows() == 0 and
      world.num_active_tags() == 0 and world.num_delayed_events() == 0 and
      world.num_flow_completions() == EXPECTED_DYNAMIC,
      f"defs={world.num_flow_defs()} pending={world.num_pending_flows()} "
      f"tags={world.num_active_tags()} events={world.num_delayed_events()} "
      f"completions={world.num_flow_completions()}")

# V5 coexistence: static flow-file flows (flow_order >= 0, global
# pendingFlowCursor window) and NPU-owned dynamic flows (flow_order == -1)
# must both drain to completion in the same run.
# Note the stop predicate: the usual network-side quiescence check
# (pending/events/tags all zero) is not enough here, because a FakeSystem NPU
# sitting in its Computing phase has its next wakeup in SystemEventQueue,
# which none of those three counters observe. Stage B needs a stop condition
# that also consults the system layer (isExistedFlow).
static_world = make_world(FLOWS)
steps, _ = run(static_world, 20000,
               lambda w: w.num_pending_flows() == 0 and
               w.num_delayed_events() == 0 and w.num_active_tags() == 0 and
               len(dynamic_records(w)) >= EXPECTED_DYNAMIC)
all_recs = [r for r in static_world.flow_completions() if r["flow_id"] >= 0]
dyn = dynamic_records(static_world)
static_recs = [r for r in all_recs if r["flow_id"] < 900000]
check("V5 static + dynamic coexistence",
      len(dyn) == EXPECTED_DYNAMIC and len(static_recs) > 0 and
      len(all_recs) == static_world.num_flow_completions() and
      static_world.num_flow_defs() == 0,
      f"static_completions={len(static_recs)} dynamic_completions={len(dyn)} "
      f"total={static_world.num_flow_completions()} steps={steps} "
      f"sim_time_ms={static_world.simulation_time():.4f}")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all cases passed")
