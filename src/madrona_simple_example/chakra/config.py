from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

from .conversion import folder_to_int_array


MAX_NPUS = 128
CHAKRA_NODES_DATA_LENGTH = 5000
PROCESS_PARAMS_LENGTH = 1000


class CommImplementation(IntEnum):
    RING = 0
    TREE = 1


@dataclass(frozen=True)
class SystemConfig:
    ring_dims: tuple[int, int, int]
    chunks_num: int
    npu_count: int
    all_reduce: CommImplementation = CommImplementation.RING
    reduce: CommImplementation = CommImplementation.RING
    all_gather: CommImplementation = CommImplementation.RING
    gather: CommImplementation = CommImplementation.RING
    scatter: CommImplementation = CommImplementation.RING
    broadcast: CommImplementation = CommImplementation.RING
    all_to_all: CommImplementation = CommImplementation.RING
    reduce_scatter: CommImplementation = CommImplementation.RING
    reduce_scatter_block: CommImplementation = CommImplementation.RING
    barrier: CommImplementation = CommImplementation.RING

    def validate(self):
        if not 0 < self.npu_count <= MAX_NPUS:
            raise ValueError(f"npu_count must be in [1, {MAX_NPUS}]")
        if np.prod(self.ring_dims, dtype=np.int64) != self.npu_count:
            raise ValueError("ring_dims product must equal npu_count")
        if self.chunks_num <= 0:
            raise ValueError("chunks_num must be positive")


def build_process_params(config):
    config.validate()
    params = np.zeros(PROCESS_PARAMS_LENGTH, dtype=np.int32)
    params[0:3] = config.ring_dims
    params[3] = config.chunks_num
    params[100] = config.npu_count
    params[200:210] = [
        int(config.all_reduce),
        int(config.reduce),
        int(config.all_gather),
        int(config.gather),
        int(config.scatter),
        int(config.broadcast),
        int(config.all_to_all),
        int(config.reduce_scatter),
        int(config.reduce_scatter_block),
        int(config.barrier),
    ]
    params[998] = 1
    params[999] = 0
    return params


def load_chakra_workload(workload, npu_count):
    if isinstance(workload, (str, Path)):
        rows = folder_to_int_array(workload, max_workers=0,
                                   show_progress=False)
    else:
        rows = workload

    if len(rows) != npu_count:
        raise ValueError(
            f"workload has {len(rows)} NPU rows, expected {npu_count}")

    encoded = np.zeros(
        (MAX_NPUS, CHAKRA_NODES_DATA_LENGTH), dtype=np.int32)
    for npu_id, row in enumerate(rows):
        row_array = np.asarray(row, dtype=np.int64).reshape(-1)
        if row_array.size > CHAKRA_NODES_DATA_LENGTH:
            raise ValueError(
                f"NPU {npu_id} encoded workload exceeds "
                f"{CHAKRA_NODES_DATA_LENGTH} integers")
        if np.any(row_array < np.iinfo(np.int32).min) or \
                np.any(row_array > np.iinfo(np.uint32).max):
            raise ValueError(f"NPU {npu_id} contains an out-of-range value")
        encoded[npu_id, :row_array.size] = row_array.astype(
            np.uint32, copy=False).view(np.int32)
    return encoded

