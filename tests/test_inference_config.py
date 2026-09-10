import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from madrona_simple_example.inference import (
    INFERENCE_CONFIG_LENGTH,
    InferenceConfig,
    load_request_trace,
    load_workload_params_table,
    pack_request_trace,
    pack_workload_params_table,
    parse_inference_stats,
)

IC_P_WORKERS_BASE = 32
IC_D_WORKERS_BASE = 64


def make_config(**overrides):
    values = {
        "enabled": True,
        "prefill_workers": (range(0, 2), range(2, 4)),
        "decode_workers": (range(4, 6), range(6, 8)),
        "p_max_batch": 4,
        "p_max_tokens": 2048,
        "d_max_batch": 8,
        "d_max_tokens": 4096,
        "num_layers": 24,
        "num_kv_heads": 4,
        "head_dim": 128,
        "bytes_per_elem": 2,
        "kv_partition_factor": 2,
        "prefill_reference_tokens": 512,
        "decode_reference_tokens": 1,
        "p_max_wait_ns": 1000,
        "d_max_wait_ns": 2000,
    }
    values.update(overrides)
    return InferenceConfig(**values)


class InferenceConfigTests(unittest.TestCase):
    def test_config_pack_schema_words(self):
        packed = make_config().pack(num_requests=3, npu_count=8)
        self.assertEqual(packed.dtype, np.int64)
        self.assertEqual(packed.shape, (INFERENCE_CONFIG_LENGTH,))
        np.testing.assert_array_equal(
            packed[:26],
            np.array([
                1, 3, 2, 2, 0, 0, 0, 0, 0, 0, 4, 2048, 8, 4096,
                24, 4, 128, 2, 2, 512, 1, 1000, 0, 2000, 0, 0,
            ], dtype=np.int64),
        )
        self.assertTrue(np.all(packed[26:IC_P_WORKERS_BASE] == 0))

    def test_config_pack_worker_lists(self):
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3), (4, 5)),
            decode_workers=((6,), (7, 8)),
        )
        packed = config.pack(
            num_requests=2, npu_count=9, workload_params_table_count=5)
        self.assertEqual(packed[2], 2)
        self.assertEqual(packed[3], 2)
        self.assertEqual(packed[22], 5)
        # Each worker occupies (rank_start, rank_count) at the pool base.
        np.testing.assert_array_equal(
            packed[IC_P_WORKERS_BASE:IC_P_WORKERS_BASE + 4],
            np.array([0, 4, 4, 2], dtype=np.int64))
        np.testing.assert_array_equal(
            packed[IC_D_WORKERS_BASE:IC_D_WORKERS_BASE + 4],
            np.array([6, 1, 7, 2], dtype=np.int64))
        self.assertTrue(np.all(packed[IC_D_WORKERS_BASE + 4:] == 0))

    def test_unequal_widths_validate(self):
        # P and D pools no longer need equal widths; workers inside a pool
        # may also differ. Same-head KV pairing requires each width to be
        # an integer multiple of num_kv_heads and of the other pool's
        # width (kv_heads=2: widths 1/2/4 are legal).
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3), (4, 5)),
            decode_workers=((6,), (7, 8)),
        )
        self.assertEqual(
            config.validate(npu_count=9),
            (0, (4, 2), 6, (1, 2)))
        # P pool wider than D pool is legal (grouped head pairing).
        InferenceConfig(
            enabled=True,
            prefill_workers=(range(0, 16),),
            decode_workers=(range(16, 24),),
        ).validate(npu_count=24)

    def test_worker_validation(self):
        cases = (
            # Empty pool.
            {"prefill_workers": ()},
            # Zero-width worker.
            {"prefill_workers": ((), (1,))},
            # Too many workers.
            {"decode_workers": tuple(
                (worker_idx,) for worker_idx in range(17))},
            # Rank width above the per-worker cap.
            {"prefill_workers": (tuple(range(17)),)},
            # Overlapping P workers.
            {"prefill_workers": ((0, 1), (1, 2))},
            # Overlapping P and D pools.
            {"decode_workers": ((1, 2),)},
            # Non-contiguous worker range.
            {"prefill_workers": ((0, 2),)},
        )
        for override in cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                config = make_config(**override)
                config.validate(npu_count=16)

    def test_ranks_must_fit_npu_count(self):
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1),),
            decode_workers=((2, 3),),
        )
        # Rank 3 is the last valid rank, so npu_count=4 still fits.
        config.validate(npu_count=4)
        config.validate(npu_count=5)
        with self.assertRaisesRegex(ValueError, "npu_count"):
            InferenceConfig(
                enabled=True,
                prefill_workers=((0, 1),),
                decode_workers=((2, 3, 4),),
            ).validate(npu_count=4)

    def test_kv_head_multiple_layout_required(self):
        # Same-head KV pairing: every P/D worker width must be an integer
        # multiple of num_kv_heads (or vice versa) and of the other pool's
        # width. kv_heads=6 breaks against width 4 (neither divides); the
        # (4 vs 8) P/D widths pair with each other but the 3-wide D
        # worker breaks the kv_heads relation.
        with self.assertRaisesRegex(ValueError, "num_kv_heads"):
            InferenceConfig(
                enabled=True,
                prefill_workers=((0, 1, 2, 3),),
                decode_workers=((4, 5),),
                num_kv_heads=6,
            ).validate(npu_count=6)
        with self.assertRaisesRegex(ValueError, "num_kv_heads"):
            InferenceConfig(
                enabled=True,
                prefill_workers=((0, 1),),
                decode_workers=((2, 3, 4),),
                num_kv_heads=2,
            ).validate(npu_count=5)
        # Widths that divide kv_heads (or vice versa) pass: width 1, 2, 4
        # against kv_heads=2; D wider than P with the 2|4 relation.
        for p_width, d_width in ((4, 2), (2, 4), (1, 4), (4, 4)):
            with self.subTest(p_width=p_width, d_width=d_width):
                InferenceConfig(
                    enabled=True,
                    prefill_workers=(tuple(range(p_width)),),
                    decode_workers=(
                        tuple(range(p_width, p_width + d_width)),),
                    num_kv_heads=2,
                ).validate(npu_count=p_width + d_width)

    def test_kv_partition_factor_must_restate_p_rank_count(self):
        # An explicit factor must equal every P worker rank_count (it
        # restates the same GQA layout); any other positive value is
        # rejected because intermediate shard counts cannot pair heads.
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3),),
            decode_workers=((4, 5),),
            num_kv_heads=2,
            kv_partition_factor=3,
        )
        with self.assertRaisesRegex(ValueError, "kv_partition_factor"):
            config.validate(npu_count=6)
        # Equal to the (single, uniform) P width: accepted.
        InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3),),
            decode_workers=((4, 5),),
            num_kv_heads=2,
            kv_partition_factor=4,
        ).validate(npu_count=6)
        # Uneven P widths with an explicit factor can never restate all
        # workers: rejected even when the factor matches one of them.
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3), (4, 5)),
            decode_workers=((6, 7),),
            num_kv_heads=2,
            kv_partition_factor=4,
        )
        with self.assertRaisesRegex(ValueError, "kv_partition_factor"):
            config.validate(npu_count=8)

    def test_kv_partition_factor_zero_is_auto_gqa_default(self):
        # 0 = auto stays the default: shards follow the GQA layout (one
        # stream per emitting P rank holding a distinct head block). MLA
        # (kv_mode=1) ignores this field and requires factor ∈ {0, 1}
        # instead; it streams one full-latent copy to every D rank.
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3),),
            decode_workers=((4, 5),),
            num_kv_heads=2,
        )
        self.assertEqual(config.kv_partition_factor, 0)
        packed = config.pack(num_requests=1, npu_count=6)
        self.assertEqual(packed[18], 0)

    def test_wait_thresholds_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "p_max_wait_ns"):
            make_config(p_max_wait_ns=-1).validate()
        with self.assertRaisesRegex(ValueError, "d_max_wait_ns"):
            make_config(d_max_wait_ns=-1).validate()

    def test_kv_mode_and_latent_dim_pack(self):
        # Slots 24/25 carry kv_mode / kv_latent_dim; the GQA defaults are
        # both 0 and leave every existing slot untouched.
        config = make_config(
            kv_mode=1, kv_latent_dim=576, kv_partition_factor=0)
        packed = config.pack(num_requests=1, npu_count=8)
        self.assertEqual(packed[24], 1)
        self.assertEqual(packed[25], 576)
        self.assertEqual(packed[:24].tolist()[-1], 2000)  # slot 23 unchanged

    def test_kv_mode_mla_requires_latent_dim(self):
        # MLA without a latent width is a missing config anchor.
        with self.assertRaisesRegex(ValueError, "kv_mode=1"):
            make_config(kv_mode=1).validate()
        with self.assertRaisesRegex(ValueError, "kv_latent_dim"):
            make_config(kv_mode=1, kv_latent_dim=0).validate()

    def test_kv_mode_gqa_rejects_latent_dim(self):
        # Mixed configuration guard: kv_latent_dim is the MLA latent
        # width and must stay 0 in GQA mode.
        with self.assertRaisesRegex(ValueError, "kv_mode=0"):
            make_config(kv_latent_dim=576).validate()

    def test_kv_mode_mla_skips_divisibility_and_factor(self):
        # MLA has no KV heads to pair: any P/D width combination is legal
        # (kv_heads=6 against p_width=4 breaks GQA but not MLA), and an
        # explicit factor restating a GQA layout is rejected.
        config = InferenceConfig(
            enabled=True,
            prefill_workers=((0, 1, 2, 3),),
            decode_workers=((4, 5),),
            num_kv_heads=6,
            kv_mode=1,
            kv_latent_dim=576,
        )
        self.assertEqual(
            config.validate(npu_count=6), (0, (4,), 4, (2,)))
        with self.assertRaisesRegex(ValueError, "kv_partition_factor"):
            make_config(
                kv_mode=1, kv_latent_dim=576, kv_partition_factor=2,
                num_kv_heads=2,
            ).validate()

    def test_kv_mode_mla_accepts_factor_one(self):
        # factor ∈ {0, 1} in MLA mode: 0 = default, 1 = explicit
        # restatement of the single latent shard.
        for factor in (0, 1):
            with self.subTest(factor=factor):
                make_config(
                    kv_mode=1, kv_latent_dim=576, kv_partition_factor=factor,
                ).validate()

    def test_kv_mode_validation(self):
        with self.assertRaisesRegex(ValueError, "kv_mode"):
            make_config(kv_mode=2).validate()
        with self.assertRaisesRegex(ValueError, "kv_latent_dim"):
            make_config(kv_latent_dim=-1).validate()

    def test_deprecated_fields_are_rejected(self):
        # The p/d route policy fields were removed (QueueDepth is the only
        # policy); passing them must fail. d_max_wait_ns is a real field
        # again (D side uses the same dual-threshold gate as P).
        for field in ("p_route_policy", "d_route_policy"):
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    make_config(**{field: 0})


class RequestTraceTests(unittest.TestCase):
    def test_list_and_array_are_sorted_and_packed(self):
        rows = [
            {"request_id": 2, "arrival_time_ns": 20,
             "prompt_len": 4, "output_len": 2},
            {"request_id": 1, "arrival_time_ns": 10,
             "prompt_len": 3, "output_len": 0},
        ]
        loaded = load_request_trace(rows)
        np.testing.assert_array_equal(
            loaded,
            np.array([[1, 10, 3, 0], [2, 20, 4, 2]], dtype=np.int64))
        packed = pack_request_trace(loaded)
        self.assertEqual(packed.shape, (1024, 4))
        np.testing.assert_array_equal(packed[:2], loaded)
        self.assertTrue(np.all(packed[2:] == 0))

    def test_json_jsonl_and_csv(self):
        rows = [
            {"request_id": 7, "arrival_time_ns": 4,
             "prompt_len": 8, "output_len": 3}
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "trace.json"
            json_path.write_text(json.dumps(rows), encoding="utf-8")
            jsonl_path = root / "trace.jsonl"
            jsonl_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            csv_path = root / "trace.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            for path in (json_path, jsonl_path, csv_path):
                with self.subTest(path=path):
                    np.testing.assert_array_equal(
                        load_request_trace(path),
                        np.array([[7, 4, 8, 3]], dtype=np.int64))

    def test_trace_validation(self):
        duplicate = [
            {"request_id": 1, "arrival_time_ns": 0,
             "prompt_len": 1, "output_len": 1},
            {"request_id": 1, "arrival_time_ns": 1,
             "prompt_len": 1, "output_len": 1},
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            load_request_trace(duplicate)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            load_request_trace([{
                "request_id": 1, "arrival_time_ns": -1,
                "prompt_len": 1, "output_len": 1,
            }])
        with self.assertRaisesRegex(ValueError, "1024"):
            load_request_trace([{
                "request_id": idx, "arrival_time_ns": idx,
                "prompt_len": 1, "output_len": 1,
            } for idx in range(1025)])


class WorkloadParamsTableTests(unittest.TestCase):
    def test_table_schema_normalization_and_pack(self):
        rows = load_workload_params_table([
            {
                "stage": "prefill",
                "node_type": "COMP_NODE",
                "batch_size": 2,
                "sequence_tokens": 8,
                "duration_ns": 1500,
                "comm_size_bytes": 0,
            },
            {
                "stage": "decode",
                "node_type": 4,
                "batch_size": -1,
                "sequence_tokens": -1,
                "duration_ns": 0,
                "comm_size_bytes": 4096,
            },
        ])
        np.testing.assert_array_equal(
            rows,
            np.array([
                [1, 1, 2, 8, 1500, 0],
                [2, 4, -1, -1, 0, 4096],
            ], dtype=np.int64))
        packed = pack_workload_params_table(rows)
        self.assertEqual(packed.shape, (1024, 6))
        np.testing.assert_array_equal(packed[:2], rows)
        self.assertTrue(np.all(packed[2:] == 0))

    def test_table_json_jsonl_and_csv(self):
        row = {
            "stage": "decode",
            "node_type": "COMM_SEND_NODE",
            "batch_size": -1,
            "sequence_tokens": 4,
            "duration_ns": 0,
            "comm_size_bytes": 128,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "table.json"
            json_path.write_text(
                json.dumps({"profiles": [row]}), encoding="utf-8")
            jsonl_path = root / "table.jsonl"
            jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            csv_path = root / "table.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            expected = np.array([[2, 2, -1, 4, 0, 128]],
                                dtype=np.int64)
            for path in (json_path, jsonl_path, csv_path):
                with self.subTest(path=path):
                    np.testing.assert_array_equal(
                        load_workload_params_table(path), expected)

    def test_table_validation(self):
        base = {
            "stage": "prefill",
            "node_type": "COMP_NODE",
            "batch_size": 1,
            "sequence_tokens": 1,
            "duration_ns": 1,
            "comm_size_bytes": 0,
        }
        for field, value in (
                ("batch_size", 0),
                ("sequence_tokens", 0),
                ("duration_ns", -1),
                ("comm_size_bytes", -1)):
            row = dict(base)
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_workload_params_table([row])


class InferenceStatsTests(unittest.TestCase):
    def test_metric_formulas(self):
        raw = np.zeros((1, 19), dtype=np.int64)
        raw[0] = [
            9, 100, 120, 200, 200, 200, 240, 300, 500,
            0, 0, 4, 4, 4096, 7, 32, 2, 2, 350,
        ]
        row = parse_inference_stats(raw)[0]
        self.assertEqual(row["ttft_ns"], 250)
        self.assertEqual(row["tpot_ns"], 50.0)
        self.assertEqual(row["e2e_ns"], 400)

    def test_zero_output_has_zero_tpot(self):
        raw = np.zeros((1, 19), dtype=np.int64)
        raw[0, 1] = 10
        raw[0, 7] = 30
        raw[0, 8] = 30
        raw[0, 18] = 30
        self.assertEqual(parse_inference_stats(raw)[0]["tpot_ns"], 0.0)


if __name__ == "__main__":
    unittest.main()
