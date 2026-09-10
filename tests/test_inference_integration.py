import os
import unittest

import numpy as np

try:
    from madrona_simple_example import GridWorld, InferenceConfig, SystemConfig
    from madrona_simple_example.chakra.conversion import node_to_int_array
    from madrona_simple_example.chakra.parser import Node
except ImportError:
    GridWorld = None


INFERENCE_KV_FLOW_BIT = 0x80000000


def inference_kv_flows(world):
    """Decode inference KV flows into (request_slot, shard, src, dst, size)."""
    flows = []
    for record in world.flow_completions():
        flow_id = record["flow_id"]
        if not flow_id & INFERENCE_KV_FLOW_BIT:
            continue
        slot = (flow_id & ~INFERENCE_KV_FLOW_BIT) >> 4
        shard = flow_id & 0xF
        flows.append((slot, shard, record["src_node"],
                      record["dst_node"], record["size_bytes"]))
    return flows


@unittest.skipIf(GridWorld is None, "Madrona extension is not built")
class InferenceIntegrationTests(unittest.TestCase):
    def _make_world(self, requests,
                    workload_params_table=None, inference_config=None,
                    npu_count=4):
        row = node_to_int_array(Node(
            id="0",
            name="forward",
            type="COMP_NODE",
            dataDeps=[],
            durationMicros=1,
            attr=[],
        ))
        network = {
            "node_ids": np.arange(npu_count + 1, dtype=np.int32),
            "node_types": np.array(
                [0] * npu_count + [1], dtype=np.int32),
            "node_port_bws": np.full(npu_count + 1, 1e8, dtype=np.float64),
            "link_srcs": np.arange(npu_count, dtype=np.int32),
            "link_dsts": np.full(npu_count, npu_count, dtype=np.int32),
            "link_delays": np.full(npu_count, 0.001, dtype=np.float64),
            "link_bandwidths": np.full(npu_count, 1e8, dtype=np.float64),
            "flow_ids": np.array([], dtype=np.int64),
            "flow_src_nodes": np.array([], dtype=np.int32),
            "flow_dst_nodes": np.array([], dtype=np.int32),
            "flow_sizes": np.array([], dtype=np.float64),
            "flow_start_times": np.array([], dtype=np.float64),
            "flow_priorities": np.array([], dtype=np.int32),
        }
        if inference_config is None:
            inference_config = InferenceConfig(
                enabled=True,
                prefill_workers=((0,), (1,)),
                decode_workers=((2,), (3,)),
                p_max_batch=4,
                p_max_tokens=1024,
                d_max_batch=4,
                d_max_tokens=1024,
                num_layers=1,
                num_kv_heads=1,
                head_dim=1,
                bytes_per_elem=1,
                kv_partition_factor=1,
                prefill_reference_tokens=1,
                decode_reference_tokens=1,
            )
        return GridWorld(
            num_worlds=1,
            start_cell=np.array([0, 0]),
            end_cells=np.array([[0, 0]], dtype=np.int32),
            rewards=np.zeros((1, 1), dtype=np.float32),
            walls=np.zeros((1, 1), dtype=bool),
            gpu_sim=os.environ.get("MADSIMPLE_TEST_GPU") == "1",
            network_inputs=network,
            system_workload=[row.copy() for _ in range(npu_count)],
            system_config=SystemConfig(
                ring_dims=(npu_count, 1, 1),
                chunks_num=1,
                npu_count=npu_count,
            ),
            inference_config=inference_config,
            request_trace=requests,
            workload_params_table=workload_params_table,
        )

    def _run_to_completion(self, world, max_steps=400):
        for _ in range(max_steps):
            world.step()
            status = world.system_status()
            self.assertFalse(status["failed"], status)
            if status["finished"]:
                return world.inference_stats()
        self.fail("inference simulation did not finish")

    def test_single_request_runs_prefill_kv_and_decode(self):
        rows = self._run_to_completion(self._make_world([{
            "request_id": 42,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 2,
        }]))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["state"], 7)
        self.assertEqual(row["output_done"], 2)
        self.assertEqual(row["kv_bytes"], 2)
        self.assertEqual(row["kv_shards_done"], row["kv_shards"])
        self.assertLessEqual(row["p_start_ns"], row["p_finish_ns"])
        self.assertLessEqual(row["p_finish_ns"], row["d_route_ns"])
        self.assertLessEqual(row["d_route_ns"], row["kv_done_ns"])
        self.assertLessEqual(row["kv_done_ns"], row["d_start_ns"])
        self.assertLessEqual(row["d_start_ns"], row["finish_ns"])

    def test_queue_depth_spreads_same_time_requests(self):
        # Queue-depth routing (the only policy) spreads two same-time
        # requests across the two idle workers: after the first request
        # lands on worker 0, its queue-depth value exceeds worker 1's.
        requests = [{
            "request_id": request_id,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 1,
        } for request_id in (1, 2)]
        rows = self._run_to_completion(self._make_world(requests))
        self.assertEqual({row["p_worker"] for row in rows}, {0, 1})
        self.assertEqual({row["d_worker"] for row in rows}, {0, 1})

    def test_queue_depth_tie_breaks_to_lowest_worker_id(self):
        # Equal queue-depth values must pick the smallest worker id
        # (strict less-than update). P side: three identical same-time
        # requests on two idle workers route 0, 1, then 0 again when the
        # pending values tie at 4 vs 4. D side: all three requests finish
        # prefill in one batch, so their KV routes run back to back and
        # produce the same 0, 1, 0 tie-break sequence.
        p_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0,), (1,)),
            decode_workers=((2,),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=1,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        d_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0,),),
            decode_workers=((2,), (3,)),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=1,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": request_id,
            "arrival_time_ns": 0,
            "prompt_len": 4,
            "output_len": 1,
        } for request_id in (1, 2, 3)]
        p_rows = self._run_to_completion(
            self._make_world(
                requests, inference_config=p_config, npu_count=3))
        self.assertEqual([row["p_worker"] for row in p_rows], [0, 1, 0])
        d_rows = self._run_to_completion(
            self._make_world(
                requests, inference_config=d_config))
        self.assertEqual([row["d_worker"] for row in d_rows], [0, 1, 0])

    def test_unequal_width_p2_d1_kv_shards_follow_p_rank_count(self):
        # P worker owns ranks 0-1 (width 2), D worker owns rank 2 (width
        # 1). kv_heads=2: each P rank holds one head, both stream to the
        # single D rank (kv_heads % d_width == 0). shards = p.rank_count,
        # per-shard bytes = 1 layer × 1 token × 2 (K,V) × 1 head × 1 dim
        # × 1 byte = 2.
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1),),
            decode_workers=((2,),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=1,
            num_kv_heads=2,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=0,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": request_id,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 1,
        } for request_id in (1, 2)]
        world = self._make_world(
            requests, inference_config=inference_config, npu_count=3)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        self.assertEqual({row["kv_shards"] for row in rows}, {2})
        self.assertEqual(
            {row["kv_shards_done"] for row in rows}, {2})
        self.assertEqual({row["kv_bytes"] for row in rows}, {2})
        # Each P rank streams its own head to the single D rank.
        expected = {
            (slot, shard, src, 2, 2.0)
            for slot in (0, 1) for shard, src in ((0, 0), (1, 1))
        }
        self.assertEqual(set(inference_kv_flows(world)), expected)

    def test_gqa_same_head_pairing_wide_p_narrow_d(self):
        # Same-head pairing (p>d): P worker with 4 ranks, D worker with 2
        # ranks, kv_heads=4 (1 head per P rank, 2 per D rank). Sending
        # rank i holds head i and pairs with D rank i//2: flows
        # (0->4), (1->4), (2->5), (3->5). Per-shard bytes = 1 layer × 1
        # token × 2 × 1 head × 1 dim × 1 byte = 2.
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3),),
            decode_workers=((4, 5),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=1,
            num_kv_heads=4,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=0,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 7,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 1,
        }]
        world = self._make_world(
            requests, inference_config=inference_config, npu_count=6)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        self.assertEqual(rows[0]["kv_shards"], 4)
        self.assertEqual(rows[0]["kv_bytes"], 2)
        expected = {
            (0, shard, shard, 4 + shard // 2, 2.0)
            for shard in range(4)
        }
        self.assertEqual(set(inference_kv_flows(world)), expected)

    def test_gqa_same_head_pairing_narrow_p_wide_d(self):
        # Same-head pairing (p<d): P worker with 2 ranks, D worker with 4
        # ranks, kv_heads=2 (1 head per P rank, a 2-rank D group per
        # head). Sending rank i holds head i and pairs with the first D
        # rank of head i's group: flows (0->2), (1->4); D ranks 3 and 5
        # never receive flow. Per-shard bytes = 1 layer × 1 token × 2 ×
        # 1 head × 1 dim × 1 byte = 2.
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1),),
            decode_workers=((2, 3, 4, 5),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=1,
            num_kv_heads=2,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=0,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 7,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 1,
        }]
        world = self._make_world(
            requests, inference_config=inference_config, npu_count=6)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        self.assertEqual(rows[0]["kv_shards"], 2)
        self.assertEqual(rows[0]["kv_bytes"], 2)
        expected = {
            (0, shard, shard, 2 + shard * 2, 2.0)
            for shard in range(2)
        }
        self.assertEqual(set(inference_kv_flows(world)), expected)

    def test_gqa_replication_silent_duplicate_ranks(self):
        # Replication dedup (p_width > kv_heads): P worker with 4 ranks,
        # D worker with 2 ranks, kv_heads=2. Copy factor r = 4/2 = 2:
        # ranks 0,2 hold heads 0,1 and stream; ranks 1,3 are replicas and
        # stay silent. Only 2 flows: (0->4), (2->5). heads_per_p_rank=1 so
        # per-shard bytes = 1 layer × 1 token × 2 × 1 head × 1 dim × 1
        # byte = 2.
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3),),
            decode_workers=((4, 5),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=1,
            num_kv_heads=2,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=0,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 7,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 1,
        }]
        world = self._make_world(
            requests, inference_config=inference_config, npu_count=6)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        self.assertEqual(rows[0]["kv_shards"], 2)
        self.assertEqual(rows[0]["kv_bytes"], 2)
        expected = {
            (0, shard, shard * 2, 4 + shard, 2.0)
            for shard in range(2)
        }
        self.assertEqual(set(inference_kv_flows(world)), expected)

    def test_gqa_per_rank_head_share_bytes(self):
        # Head-share byte accounting: P and D both 2 ranks with
        # kv_heads=2 (1 head per rank). Per-shard bytes = layers × tokens
        # × 2 (K,V) × heads_per_p_rank × head_dim × bytes_per_elem =
        # 2 × 3 × 2 × 1 × 4 × 2 = 96, and shards sum to the full
        # kv_bytes... except here shards=2 equals the head count so each
        # shard carries one head's share (96) and kv_bytes reports the
        # per-rank share (heads_per_p_rank × ... = 96).
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1),),
            decode_workers=((2, 3),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=2,
            num_kv_heads=2,
            head_dim=4,
            bytes_per_elem=2,
            kv_partition_factor=0,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 7,
            "arrival_time_ns": 0,
            "prompt_len": 3,
            "output_len": 1,
        }]
        world = self._make_world(
            requests, inference_config=inference_config, npu_count=4)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        self.assertEqual(rows[0]["kv_shards"], 2)
        expected = {
            (0, shard, shard, 2 + shard, 96.0)
            for shard in range(2)
        }
        self.assertEqual(set(inference_kv_flows(world)), expected)

    def test_mla_point_to_point_wide_p_narrow_d(self):
        # MLA (kv_mode=1) point-to-point replication (form B): the P
        # group's first rank (p.rank_start) streams one FULL latent copy
        # to every rank of the routed D worker — shards = d_width. Each
        # flow size = layers × prompt_len × kv_latent_dim ×
        # bytes_per_elem = 2 × 3 × 6 × 2 = 72 (no splitting). No
        # divisibility requirement applies — kv_heads=6 against p_width=4
        # would be InvalidConfig in GQA mode but is legal here.
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3),),
            decode_workers=((4, 5),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=2,
            num_kv_heads=6,
            head_dim=1,
            bytes_per_elem=2,
            kv_partition_factor=0,
            kv_mode=1,
            kv_latent_dim=6,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 7,
            "arrival_time_ns": 0,
            "prompt_len": 3,
            "output_len": 1,
        }]
        world = self._make_world(
            requests, inference_config=inference_config, npu_count=6)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        self.assertEqual(rows[0]["kv_shards"], 2)
        self.assertEqual(rows[0]["kv_shards_done"], 2)
        self.assertEqual(rows[0]["kv_bytes"], 72)
        flows = inference_kv_flows(world)
        self.assertEqual(set(flows),
                          {(0, 0, 0, 4, 72.0), (0, 1, 0, 5, 72.0)})
        # kv_done fires only after the last of the d_width streams lands.
        latest_flow_end_ns = max(
            record["end_time_ms"] * 1e6
            for record in world.flow_completions()
            if record["flow_id"] & INFERENCE_KV_FLOW_BIT)
        self.assertGreaterEqual(rows[0]["kv_done_ns"] + 1,
                                latest_flow_end_ns)

    def test_mla_point_to_point_every_d_rank_gets_full_latent(self):
        # MLA streams one full-latent flow per D rank (d_width=4 here):
        # p_width=2/d_width=4 produces 4 flows, all sourced from
        # p.rank_start=0 and targeting d.rank_start..+3, each carrying the
        # complete 72B latent (never split across the D ranks).
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1),),
            decode_workers=((2, 3, 4, 5),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=2,
            num_kv_heads=6,
            head_dim=1,
            bytes_per_elem=2,
            kv_partition_factor=0,
            kv_mode=1,
            kv_latent_dim=6,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 7,
            "arrival_time_ns": 0,
            "prompt_len": 3,
            "output_len": 1,
        }]
        world = self._make_world(
            requests, inference_config=inference_config, npu_count=6)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        self.assertEqual(rows[0]["kv_shards"], 4)
        self.assertEqual(rows[0]["kv_shards_done"], 4)
        self.assertEqual(rows[0]["kv_bytes"], 72)
        self.assertEqual(set(inference_kv_flows(world)),
                          {(0, shard, 0, 2 + shard, 72.0)
                           for shard in range(4)})
        latest_flow_end_ns = max(
            record["end_time_ms"] * 1e6
            for record in world.flow_completions()
            if record["flow_id"] & INFERENCE_KV_FLOW_BIT)
        self.assertGreaterEqual(rows[0]["kv_done_ns"] + 1,
                                latest_flow_end_ns)

    def test_mla_multiple_requests_each_stream_to_every_d_rank(self):
        # Two routed requests under MLA: each request streams its own
        # d_width full-latent flows (per-request shard s -> d.rank_start+s,
        # slot-scoped flow ids; all flows originate at p.rank_start=0).
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3),),
            decode_workers=((4, 5),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            num_layers=2,
            num_kv_heads=6,
            head_dim=1,
            bytes_per_elem=2,
            kv_partition_factor=0,
            kv_mode=1,
            kv_latent_dim=6,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": request_id,
            "arrival_time_ns": 0,
            "prompt_len": 3,
            "output_len": 1,
        } for request_id in (1, 2)]
        world = self._make_world(
            requests, inference_config=inference_config, npu_count=6)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        self.assertEqual({row["kv_shards"] for row in rows}, {2})
        self.assertEqual(
            set(inference_kv_flows(world)),
            {(slot, shard, 0, 4 + shard, 72.0)
             for slot in (0, 1) for shard in (0, 1)})

    def test_prefill_starts_on_max_wait_timeout_without_full_batch(self):
        # p_max_batch=4 but only 2 requests: prefill must NOT launch at
        # arrival; it waits until the head request has waited
        # p_max_wait_ns, then starts even though the batch is not full.
        inference_config = InferenceConfig(
            enabled=True,
            prefill_workers=((0,),),
            decode_workers=((1,),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            p_max_wait_ns=3000,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=1,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 1,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 1,
        }, {
            "request_id": 2,
            "arrival_time_ns": 500,
            "prompt_len": 1,
            "output_len": 1,
        }]
        world = self._make_world(
            requests, inference_config=inference_config)
        rows = self._run_to_completion(world)
        self.assertEqual({row["state"] for row in rows}, {7})
        for row in rows:
            self.assertGreaterEqual(
                row["p_start_ns"], row["arrival_time_ns"] + 3000 - 500,
                "prefill started before the max_wait threshold")

    def test_decode_continuous_batching_admits_without_delay(self):
        # d_max_wait_ns defaults to 0 (plain continuous batching): while a
        # long decode keeps the worker busy, a later KV-ready request is
        # admitted (and starts decoding) at the very next generation
        # boundary, and the already-active request's own schedule is not
        # delayed by the admission.
        def build():
            return InferenceConfig(
                enabled=True,
                prefill_workers=((0,),),
                decode_workers=((1,),),
                p_max_batch=4,
                p_max_tokens=64,
                d_max_batch=4,
                d_max_tokens=64,
                num_layers=1,
                num_kv_heads=1,
                head_dim=1,
                bytes_per_elem=1,
                kv_partition_factor=1,
                prefill_reference_tokens=1,
                decode_reference_tokens=1,
            )

        requests = [{
            "request_id": 1,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 10,
        }, {
            "request_id": 2,
            "arrival_time_ns": 5000,
            "prompt_len": 1,
            "output_len": 1,
        }]
        rows = self._run_to_completion(
            self._make_world(requests, inference_config=build()))
        self.assertEqual({row["state"] for row in rows}, {7})
        by_id = {row["request_id"]: row for row in rows}
        # The queued request is admitted promptly: it starts decoding on
        # the first generation boundary after its KV transfer completes
        # instead of waiting for a batch-assembly timeout.
        gap = by_id[2]["d_start_ns"] - by_id[2]["kv_done_ns"]
        self.assertGreaterEqual(gap, 0)
        self.assertLess(gap, 5000, "queued request was not admitted "
                                   "immediately at a decode boundary")
        # The active request is not delayed by the admission: its decode
        # cadence (first-generation latency and per-generation pace) is
        # identical to a solo run without the second request. Absolute
        # wall times may differ between the two runs because the extra
        # arrival shifts the engine's event-driven clock, but no
        # generation may stretch once decoding has started.
        solo = self._run_to_completion(
            self._make_world([requests[0]], inference_config=build()))
        self.assertEqual(solo[0]["first_token_ns"] - solo[0]["d_start_ns"],
                         by_id[1]["first_token_ns"] - by_id[1]["d_start_ns"])
        self.assertEqual(solo[0]["finish_ns"] - solo[0]["first_token_ns"],
                         by_id[1]["finish_ns"] - by_id[1]["first_token_ns"])

    def test_decode_queue_empty_runs_generations_without_hold(self):
        # Dual-threshold D gate, queue-empty branch: once the batch is
        # admitted, the queue is empty and every subsequent generation must
        # launch immediately ("有东西就跑") even with a huge d_max_wait_ns.
        # Only the first generation (queue non-empty, batch not full) pays
        # the timeout. Checked via generation rhythm: the first-generation
        # latency and the post-first-token span must match a run with
        # d_max_wait_ns=0 exactly.
        def build(max_wait):
            return InferenceConfig(
                enabled=True,
                prefill_workers=((0,),),
                decode_workers=((1,),),
                p_max_batch=4,
                p_max_tokens=64,
                d_max_batch=4,
                d_max_tokens=64,
                d_max_wait_ns=max_wait,
                num_layers=1,
                num_kv_heads=1,
                head_dim=1,
                bytes_per_elem=1,
                kv_partition_factor=1,
                prefill_reference_tokens=1,
                decode_reference_tokens=1,
            )

        requests = [{
            "request_id": 1,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 3,
        }, {
            "request_id": 2,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 3,
        }]
        held = self._run_to_completion(
            self._make_world(requests, inference_config=build(60000)))
        immediate = self._run_to_completion(
            self._make_world(requests, inference_config=build(0)))
        held_by_id = {row["request_id"]: row for row in held}
        immediate_by_id = {row["request_id"]: row for row in immediate}
        self.assertEqual({row["state"] for row in held}, {7})
        for request_id in (1, 2):
            held_row = held_by_id[request_id]
            immediate_row = immediate_by_id[request_id]
            # First generation was held until the head request's timeout
            # (batch of 2 never fills d_max_batch=4).
            self.assertGreaterEqual(
                held_row["d_start_ns"] - held_row["kv_done_ns"], 50000,
                "first generation was not held for d_max_wait_ns")
            self.assertLess(held_row["d_start_ns"] - held_row["kv_done_ns"],
                            70000, "generation did not start at the timeout")
            # Queue empty afterwards: identical generation rhythm to the
            # d_max_wait_ns=0 run (same first-gen duration, same per-gen
            # pace). A per-generation hold would blow these up.
            self.assertEqual(
                held_row["first_token_ns"] - held_row["d_start_ns"],
                immediate_row["first_token_ns"] - immediate_row["d_start_ns"])
            self.assertEqual(
                held_row["finish_ns"] - held_row["first_token_ns"],
                immediate_row["finish_ns"] - immediate_row["first_token_ns"])

    def test_decode_holds_until_max_wait_when_batch_not_full(self):
        # Dual-threshold D gate, hold branch: while r1 keeps decoding, a
        # later KV-ready request sits in the queue (active + queue = 2 <
        # d_max_batch) and the whole worker holds its next generation until
        # the head request's wait since t_kv_done_ns reaches d_max_wait_ns;
        # then both decode together in one merged batch.
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0,),),
            decode_workers=((1,),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            d_max_wait_ns=3000,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=1,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 1,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 10,
        }, {
            "request_id": 2,
            "arrival_time_ns": 10000,
            "prompt_len": 1,
            "output_len": 1,
        }]
        rows = self._run_to_completion(
            self._make_world(requests, inference_config=config))
        self.assertEqual({row["state"] for row in rows}, {7})
        by_id = {row["request_id"]: row for row in rows}
        gap = by_id[2]["d_start_ns"] - by_id[2]["kv_done_ns"]
        # Held: not admitted before the head waited d_max_wait_ns, and the
        # launch came from the timeout, not from a full batch.
        self.assertGreaterEqual(gap, 3000,
                                "queued request admitted before max_wait")
        self.assertLess(gap, 23000, "timeout did not trigger the launch")
        # The hold ended with a merged launch while r1 was still decoding.
        self.assertLess(by_id[2]["d_start_ns"], by_id[1]["finish_ns"])

    def test_decode_starts_on_max_wait_timeout_without_full_batch(self):
        # Dual-threshold D gate, timeout branch: a solo request can never
        # fill d_max_batch=4, so its first decode generation must not start
        # before t_kv_done_ns + d_max_wait_ns, and the timeout itself must
        # trigger the launch.
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0,),),
            decode_workers=((1,),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=4,
            d_max_tokens=64,
            d_max_wait_ns=3000,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=1,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 1,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 2,
        }]
        rows = self._run_to_completion(
            self._make_world(requests, inference_config=config))
        self.assertEqual({row["state"] for row in rows}, {7})
        row = rows[0]
        gap = row["d_start_ns"] - row["kv_done_ns"]
        self.assertGreaterEqual(gap, 3000,
                                "decode started before the max_wait threshold")
        self.assertLess(gap, 23000, "timeout did not trigger the launch")

    def test_decode_launches_immediately_when_batch_full(self):
        # Dual-threshold D gate, batch-full branch: with d_max_wait_ns huge
        # (timeout can never be the trigger), the queued request still
        # launches as soon as active + queue fills d_max_batch. r1 waits in
        # the queue until r2's KV lands and fills the batch, then both
        # decode together immediately.
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0,),),
            decode_workers=((1,),),
            p_max_batch=4,
            p_max_tokens=64,
            d_max_batch=2,
            d_max_tokens=64,
            d_max_wait_ns=1000000,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            bytes_per_elem=1,
            kv_partition_factor=1,
            prefill_reference_tokens=1,
            decode_reference_tokens=1,
        )
        requests = [{
            "request_id": 1,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 10,
        }, {
            "request_id": 2,
            "arrival_time_ns": 5000,
            "prompt_len": 1,
            "output_len": 10,
        }]
        rows = self._run_to_completion(
            self._make_world(requests, inference_config=config))
        self.assertEqual({row["state"] for row in rows}, {7})
        by_id = {row["request_id"]: row for row in rows}
        for request_id in (1, 2):
            row = by_id[request_id]
            gap = row["d_start_ns"] - row["kv_done_ns"]
            self.assertLess(
                gap, 20000,
                "full batch did not launch before the max_wait timeout")
        # Both requests were admitted into the same first generation.
        self.assertEqual(by_id[1]["first_token_ns"],
                         by_id[2]["first_token_ns"])

    def test_profile_exact_match_precedes_wildcard_and_changes_timing(self):
        requests = [{
            "request_id": 8,
            "arrival_time_ns": 0,
            "prompt_len": 1,
            "output_len": 1,
        }]
        baseline = self._run_to_completion(
            self._make_world(requests))[0]
        profiles = [
            {
                "stage": "prefill",
                "node_type": "COMP_NODE",
                "batch_size": -1,
                "sequence_tokens": -1,
                "duration_ns": 2000,
                "comm_size_bytes": 0,
            },
            {
                "stage": "prefill",
                "node_type": "COMP_NODE",
                "batch_size": 1,
                "sequence_tokens": 1,
                "duration_ns": 9000,
                "comm_size_bytes": 0,
            },
            {
                "stage": "decode",
                "node_type": "COMP_NODE",
                "batch_size": -1,
                "sequence_tokens": -1,
                "duration_ns": 7000,
                "comm_size_bytes": 0,
            },
        ]
        profiled = self._run_to_completion(
            self._make_world(
                requests, workload_params_table=profiles))[0]
        baseline_prefill = (
            baseline["p_finish_ns"] - baseline["p_start_ns"])
        profiled_prefill = (
            profiled["p_finish_ns"] - profiled["p_start_ns"])
        baseline_decode = (
            baseline["finish_ns"] - baseline["d_start_ns"])
        profiled_decode = (
            profiled["finish_ns"] - profiled["d_start_ns"])
        self.assertEqual(profiled_prefill, 9000)
        self.assertEqual(profiled_decode, 7000)
        self.assertGreater(profiled_prefill, baseline_prefill)
        self.assertGreater(profiled_decode, baseline_decode)


if __name__ == "__main__":
    unittest.main()
