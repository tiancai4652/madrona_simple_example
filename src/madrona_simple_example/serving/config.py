"""PD inference configuration and the fixed C++ tensor schema."""

from dataclasses import dataclass

import numpy as np


INFERENCE_CONFIG_LENGTH = 256
MAX_SERVING_WORKERS = 16
MAX_SERVING_WORKER_RANKS = 16
MAX_SERVING_BATCH = 128

IC_ENABLED = 0
IC_NUM_REQUESTS = 1
IC_NUM_P_WORKERS = 2
IC_NUM_D_WORKERS = 3
IC_P_ROUTE_POLICY = 8  # deprecated: routing is fixed to QueueDepth
IC_D_ROUTE_POLICY = 9  # deprecated: routing is fixed to QueueDepth
IC_P_MAX_BATCH = 10
IC_P_MAX_BATCH_TOKENS = 11
IC_D_MAX_BATCH = 12
IC_D_MAX_BATCH_TOKENS = 13
IC_NUM_LAYERS = 14
IC_NUM_KV_HEADS = 15
IC_HEAD_DIM = 16
IC_BYTES_PER_ELEM = 17
IC_KV_PARTITION_FACTOR = 18
IC_PREFILL_REFERENCE_TOKENS = 19
IC_DECODE_REFERENCE_TOKENS = 20
IC_P_MAX_WAIT_NS = 21
IC_WORKLOAD_PARAMS_TABLE_COUNT = 22
IC_D_MAX_WAIT_NS = 23
IC_KV_MODE = 24
IC_KV_LATENT_DIM = 25
IC_P_WORKERS_BASE = 32
IC_D_WORKERS_BASE = 64

INFERENCE_CONFIG_WORDS = (
    "enabled",
    "num_requests",
    "num_p_workers",
    "num_d_workers",
    "p_rank_start (deprecated)",
    "p_ranks_per_worker (deprecated)",
    "d_rank_start (deprecated)",
    "d_ranks_per_worker (deprecated)",
    "p_policy (deprecated: fixed QueueDepth)",
    "d_policy (deprecated: fixed QueueDepth)",
    "p_max_batch",
    "p_max_tokens",
    "d_max_batch",
    "d_max_tokens",
    "num_layers",
    "num_kv_heads",
    "head_dim",
    "bytes_per_elem",
    "kv_partition",
    "prefill_ref",
    "decode_ref",
    "p_max_wait_ns",
    "workload_params_table_count",
    "d_max_wait_ns",
    "kv_mode",
    "kv_latent_dim",
)


def _normalize_workers(workers, name):
    try:
        normalized = tuple(tuple(int(rank) for rank in worker)
                           for worker in workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a sequence of rank ranges") from exc
    return normalized


@dataclass(frozen=True)
class InferenceConfig:
    enabled: bool
    prefill_workers: tuple[tuple[int, ...], ...]
    decode_workers: tuple[tuple[int, ...], ...]
    p_max_batch: int = 16
    p_max_tokens: int = 8192
    d_max_batch: int = 16
    d_max_tokens: int = 32768
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    bytes_per_elem: int = 2
    # KV shard layout: 0 = auto (the GQA layout — each P worker rank
    # streams the KV of the contiguous head block it holds to the D rank
    # owning those heads, replicated ranks stay silent). A positive value
    # must restate that same per-rank GQA layout (== every prefill
    # worker's rank_count); any other override is rejected because no
    # intermediate shard count pairs heads cleanly. Only meaningful in
    # GQA mode (kv_mode == 0); MLA ignores it (factor must be 0 or 1).
    kv_partition_factor: int = 0
    prefill_reference_tokens: int = 1
    decode_reference_tokens: int = 1
    p_max_wait_ns: int = 0
    # D-side batch-assembly timeout: with a queued request, a new decode
    # generation starts only when d_max_wait_ns == 0 (plain continuous
    # batching), the batch would be full, or the head request has waited
    # at least d_max_wait_ns since its KV transfer completed. 0 disables
    # the wait gate (previous pure continuous batching behavior).
    d_max_wait_ns: int = 0
    # KV cache layout mode: 0 = GQA (per-head-block sharding across P/D
    # ranks, kv_partition_factor describes the shard layout) — the default
    # with unchanged legacy behavior; 1 = MLA (Multi-head Latent
    # Attention, e.g. DeepSeek-V2/V3/R1): every P rank holds the same
    # compressed latent, so prefill streams one full latent from P0 to
    # each D rank of the routed worker (shards == d_width, point-to-point
    # over the scaleout network). No divisibility requirements apply in
    # MLA mode.
    kv_mode: int = 0
    # Merged MLA latent width per layer/token = kv_lora_rank +
    # qk_rope_head_dim (DeepSeek-V3: 512 + 64 = 576). Must be 0 in GQA
    # mode; must be positive in MLA mode (kv_bytes = layers × prompt_len
    # × kv_latent_dim × bytes_per_elem).
    kv_latent_dim: int = 0

    def __post_init__(self):
        object.__setattr__(
            self, "prefill_workers",
            _normalize_workers(self.prefill_workers, "prefill_workers"))
        object.__setattr__(
            self, "decode_workers",
            _normalize_workers(self.decode_workers, "decode_workers"))

    def validate(self, npu_count=None):
        """Validate per-worker rank layout and global batch constraints."""
        p_start, p_widths = _validate_pool(
            self.prefill_workers, "prefill_workers")
        d_start, d_widths = _validate_pool(
            self.decode_workers, "decode_workers")

        p_ranks = {rank for worker in self.prefill_workers for rank in worker}
        d_ranks = {rank for worker in self.decode_workers for rank in worker}
        if p_ranks & d_ranks:
            raise ValueError("prefill and decode worker ranks must not overlap")
        if npu_count is not None:
            if npu_count <= 0:
                raise ValueError("npu_count must be positive")
            invalid = sorted(
                rank for rank in p_ranks | d_ranks if rank >= npu_count)
            if invalid:
                raise ValueError(
                    f"serving worker ranks exceed npu_count: {invalid[:8]}")

        if not 0 < self.p_max_batch <= MAX_SERVING_BATCH:
            raise ValueError(
                f"p_max_batch must be in [1, {MAX_SERVING_BATCH}]")
        if not 0 < self.d_max_batch <= MAX_SERVING_BATCH:
            raise ValueError(
                f"d_max_batch must be in [1, {MAX_SERVING_BATCH}]")
        for name in (
                "p_max_tokens", "d_max_tokens", "num_layers",
                "num_kv_heads", "head_dim", "bytes_per_elem",
                "prefill_reference_tokens", "decode_reference_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kv_mode not in (0, 1):
            raise ValueError("kv_mode must be 0 (gqa) or 1 (mla)")
        if self.kv_latent_dim < 0:
            raise ValueError("kv_latent_dim must be non-negative")
        # Mode-specific semantics are mutually exclusive to prevent mixed
        # configurations: MLA needs the latent width, GQA must not carry
        # one (a nonzero kv_latent_dim in GQA mode is almost certainly a
        # misconfigured MLA intent).
        if self.kv_mode == 1 and self.kv_latent_dim == 0:
            raise ValueError(
                "kv_mode=1 (mla) requires kv_latent_dim > 0 (the merged "
                "kv_lora_rank + qk_rope_head_dim width); set kv_latent_dim "
                "or use kv_mode=0 (gqa)")
        if self.kv_mode == 0 and self.kv_latent_dim != 0:
            raise ValueError(
                "kv_mode=0 (gqa) requires kv_latent_dim == 0; kv_latent_dim "
                "is the MLA latent width and only applies in kv_mode=1 "
                "(mla)")
        if self.kv_mode == 1:
            # MLA: the latent is a single shared vector per layer/token —
            # there are no KV heads to pair, so the GQA divisibility
            # matrix and the kv_partition_factor layout do not apply.
            # factor ∈ {0, 1} only (any other override is a GQA layout
            # statement, meaningless in MLA mode).
            if self.kv_partition_factor > 1:
                raise ValueError(
                    "kv_partition_factor must be 0 or 1 in kv_mode=1 "
                    "(mla); MLA streams one full latent to every D rank "
                    "(shards == d_width) over the scaleout network")
            return p_start, p_widths, d_start, d_widths
        kv_heads = self.num_kv_heads
        for name, widths in (("prefill", p_widths), ("decode", d_widths)):
            for worker_idx, width in enumerate(widths):
                if kv_heads % width != 0 and width % kv_heads != 0:
                    raise ValueError(
                        f"{name}_workers[{worker_idx}] rank count {width} "
                        f"must form an integer multiple relationship with "
                        f"num_kv_heads {kv_heads} (same-head KV pairing)")
        for p_idx, p_width in enumerate(p_widths):
            for d_idx, d_width in enumerate(d_widths):
                if d_width % p_width != 0 and p_width % d_width != 0:
                    raise ValueError(
                        f"prefill_workers[{p_idx}] rank count {p_width} and "
                        f"decode_workers[{d_idx}] rank count {d_width} must "
                        f"form an integer multiple relationship "
                        f"(same-head KV pairing)")
        if self.kv_partition_factor > 0 and (
                len(set(p_widths)) > 1 or
                self.kv_partition_factor != p_widths[0]):
            raise ValueError(
                "kv_partition_factor must be 0 (auto, GQA: one shard per P "
                "worker rank, same-head pairing) or equal to every prefill "
                "worker rank count (restating the same GQA layout); other "
                "overrides cannot pair heads cleanly (MLA uses kv_mode=1 "
                "and ignores this field)")
        if self.p_max_wait_ns < 0:
            raise ValueError("p_max_wait_ns must be non-negative")
        if self.d_max_wait_ns < 0:
            raise ValueError("d_max_wait_ns must be non-negative")
        return p_start, p_widths, d_start, d_widths

    def pack(self, num_requests, npu_count=None,
             workload_params_table_count=0):
        """Pack schema words 0..95 into an int64[256] C++ config row."""
        if not 0 <= num_requests <= 1024:
            raise ValueError("num_requests must be in [0, 1024]")
        if not 0 <= workload_params_table_count <= 1024:
            raise ValueError(
                "workload_params_table_count must be in [0, 1024]")
        p_start, p_widths, d_start, d_widths = self.validate(npu_count)
        values = (
            int(self.enabled),
            num_requests,
            len(self.prefill_workers),
            len(self.decode_workers),
            0,  # deprecated p_rank_start
            0,  # deprecated p_ranks_per_worker
            0,  # deprecated d_rank_start
            0,  # deprecated d_ranks_per_worker
            0,  # deprecated p_route_policy (fixed QueueDepth)
            0,  # deprecated d_route_policy (fixed QueueDepth)
            self.p_max_batch,
            self.p_max_tokens,
            self.d_max_batch,
            self.d_max_tokens,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.bytes_per_elem,
            self.kv_partition_factor,
            self.prefill_reference_tokens,
            self.decode_reference_tokens,
            self.p_max_wait_ns,
            workload_params_table_count,
            self.d_max_wait_ns,
            self.kv_mode,
            self.kv_latent_dim,
        )
        packed = np.zeros(INFERENCE_CONFIG_LENGTH, dtype=np.int64)
        packed[:len(values)] = values
        worker_start = IC_P_WORKERS_BASE
        for worker in self.prefill_workers:
            packed[worker_start] = worker[0]
            packed[worker_start + 1] = len(worker)
            worker_start += 2
        worker_start = IC_D_WORKERS_BASE
        for worker in self.decode_workers:
            packed[worker_start] = worker[0]
            packed[worker_start + 1] = len(worker)
            worker_start += 2
        return packed

# Backwards-compatible alias (kept for one release; remove after tests
# and scripts migrate).
ServingConfig = InferenceConfig


def _validate_pool(workers, name):
    if not workers:
        raise ValueError(f"{name} must contain at least one worker")
    if len(workers) > MAX_SERVING_WORKERS:
        raise ValueError(
            f"{name} exceeds the {MAX_SERVING_WORKERS} worker limit")

    widths = []
    expected_start = workers[0][0] if workers[0] else -1
    if expected_start < 0:
        raise ValueError(f"{name} ranks must be non-negative")
    pool_start = expected_start
    for worker_idx, worker in enumerate(workers):
        width = len(worker)
        if not 0 < width <= MAX_SERVING_WORKER_RANKS:
            raise ValueError(
                f"{name}[{worker_idx}] rank count must be in [1, "
                f"{MAX_SERVING_WORKER_RANKS}]")
        expected = tuple(range(expected_start, expected_start + width))
        if worker != expected:
            raise ValueError(
                f"{name}[{worker_idx}] must be the contiguous range "
                f"{expected_start}-{expected_start + width - 1}")
        expected_start += width
        widths.append(width)
    return pool_start, tuple(widths)
