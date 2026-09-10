"""Request trace loading, tensor packing, and inference statistics parsing."""

import csv
import json
from pathlib import Path

import numpy as np


REQUEST_FIELDS = (
    "request_id",
    "arrival_time_ns",
    "prompt_len",
    "output_len",
)
MAX_INFERENCE_REQUESTS = 1024
INFERENCE_STATS_FIELDS = 19
MAX_WORKLOAD_PARAMS_TABLE_ENTRIES = 1024
WORKLOAD_PARAMS_TABLE_FIELDS = (
    "stage",
    "node_type",
    "batch_size",
    "sequence_tokens",
    "duration_ns",
    "comm_size_bytes",
)
_PROFILE_STAGE_VALUES = {"prefill": 1, "decode": 2}
_PROFILE_NODE_TYPE_VALUES = {
    "COMP_NODE": 1,
    "COMM_SEND_NODE": 2,
    "COMM_RECV_NODE": 3,
    "COMM_COLL_NODE": 4,
}

_RAW_STATS_FIELDS = (
    "request_id",
    "arrival_time_ns",
    "p_start_ns",
    "p_finish_ns",
    "d_route_ns",
    "kv_start_ns",
    "kv_done_ns",
    "d_start_ns",
    "finish_ns",
    "p_worker",
    "d_worker",
    "output_len",
    "output_done",
    "kv_bytes",
    "state",
    "prompt_len",
    "kv_shards",
    "kv_shards_done",
    "first_token_ns",
)
INFERENCE_STATS_OUTPUT_FIELDS = _RAW_STATS_FIELDS + (
    "ttft_ns",
    "tpot_ns",
    "e2e_ns",
)


def load_request_trace(source):
    """Load and validate a request trace, returning sorted int64[N,4]."""
    if isinstance(source, np.ndarray):
        rows = _rows_from_array(source)
    elif isinstance(source, (str, Path)):
        rows = _load_trace_path(Path(source))
    elif isinstance(source, (list, tuple)):
        rows = list(source)
    else:
        raise TypeError(
            "request trace must be a list of dicts, numpy array, or path")

    if len(rows) > MAX_INFERENCE_REQUESTS:
        raise ValueError(
            f"request trace exceeds {MAX_INFERENCE_REQUESTS} requests")

    normalized = [_normalize_request(row, idx)
                  for idx, row in enumerate(rows)]
    request_ids = [row["request_id"] for row in normalized]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_id values must be unique")
    normalized.sort(key=lambda row: (row["arrival_time_ns"],
                                     row["request_id"]))

    result = np.zeros((len(normalized), len(REQUEST_FIELDS)), dtype=np.int64)
    for idx, row in enumerate(normalized):
        result[idx] = [row[field] for field in REQUEST_FIELDS]
    return result


def pack_request_trace(source):
    """Pack a trace into the fixed int64[1024,4] C++ input tensor."""
    trace = load_request_trace(source)
    packed = np.zeros(
        (MAX_INFERENCE_REQUESTS, len(REQUEST_FIELDS)), dtype=np.int64)
    packed[:len(trace)] = trace
    return packed


def load_workload_params_table(source):
    """Load and validate workload params table rows, returning int64[N,6]."""
    if source is None:
        rows = []
    elif isinstance(source, np.ndarray):
        array = np.asarray(source)
        if array.ndim != 2 or array.shape[1] != len(WORKLOAD_PARAMS_TABLE_FIELDS):
            raise ValueError("workload params table array must have shape [N, 6]")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("workload params table array must contain integers")
        rows = [
            dict(zip(WORKLOAD_PARAMS_TABLE_FIELDS, (int(value) for value in row)))
            for row in array
        ]
    elif isinstance(source, (str, Path)):
        rows = _load_workload_params_table_path(Path(source))
    elif isinstance(source, (list, tuple)):
        rows = list(source)
    else:
        raise TypeError(
            "workload params table must be a list of dicts, numpy array, path, or None")

    if len(rows) > MAX_WORKLOAD_PARAMS_TABLE_ENTRIES:
        raise ValueError(
            f"workload params table exceeds {MAX_WORKLOAD_PARAMS_TABLE_ENTRIES} entries")
    normalized = [_normalize_workload_params_table_row(row, idx)
                  for idx, row in enumerate(rows)]
    result = np.zeros((len(normalized), len(WORKLOAD_PARAMS_TABLE_FIELDS)),
                      dtype=np.int64)
    for idx, row in enumerate(normalized):
        result[idx] = [row[field] for field in WORKLOAD_PARAMS_TABLE_FIELDS]
    return result


def pack_workload_params_table(source):
    """Pack rows into the fixed int64[1024,6] C++ input tensor."""
    profiles = load_workload_params_table(source)
    packed = np.zeros(
        (MAX_WORKLOAD_PARAMS_TABLE_ENTRIES, len(WORKLOAD_PARAMS_TABLE_FIELDS)), dtype=np.int64)
    packed[:len(profiles)] = profiles
    return packed


def parse_inference_stats(stats, num_requests=None):
    """Convert an int64[N,18] stats tensor to dictionaries with metrics."""
    array = np.asarray(stats)
    if array.ndim != 2 or array.shape[1] != INFERENCE_STATS_FIELDS:
        raise ValueError(
            f"inference stats must have shape [N, {INFERENCE_STATS_FIELDS}]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("inference stats must contain integers")
    if num_requests is None:
        num_requests = array.shape[0]
    if not 0 <= num_requests <= array.shape[0]:
        raise ValueError("num_requests is outside the stats tensor")

    parsed = []
    for raw in array[:num_requests].astype(np.int64, copy=False):
        row = {name: int(value)
               for name, value in zip(_RAW_STATS_FIELDS, raw)}
        row["ttft_ns"] = (
            row["first_token_ns"] - row["arrival_time_ns"])
        row["tpot_ns"] = (
            (row["finish_ns"] - row["first_token_ns"]) /
            (row["output_len"] - 1)
            if row["output_len"] > 1 else 0.0
        )
        row["e2e_ns"] = row["finish_ns"] - row["arrival_time_ns"]
        parsed.append(row)
    return parsed


def _rows_from_array(array):
    if array.ndim != 2 or array.shape[1] != len(REQUEST_FIELDS):
        raise ValueError("request trace array must have shape [N, 4]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("request trace array must contain integers")
    return [
        dict(zip(REQUEST_FIELDS, (int(value) for value in row)))
        for row in array
    ]


def _load_trace_path(path):
    if not path.is_file():
        raise FileNotFoundError(f"request trace does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if isinstance(value, dict) and "requests" in value:
            value = value["requests"]
        elif isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("JSON request trace must contain a list")
        return value
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL line {line_number} is not an object")
                rows.append(value)
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    raise ValueError(
        f"unsupported request trace format {suffix!r}; "
        "expected .json, .jsonl, .ndjson, or .csv")


def _load_workload_params_table_path(path):
    if not path.is_file():
        raise FileNotFoundError(f"workload params table does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if isinstance(value, dict) and "profiles" in value:
            value = value["profiles"]
        elif isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("JSON workload params table must contain a list")
        return value
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL line {line_number} is not an object")
                rows.append(value)
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    raise ValueError(
        f"unsupported workload params table format {suffix!r}; "
        "expected .json, .jsonl, .ndjson, or .csv")


def _normalize_workload_params_table_row(row, index):
    if not isinstance(row, dict):
        raise ValueError(f"workload params table row {index} must be a dictionary")
    missing = [field for field in WORKLOAD_PARAMS_TABLE_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"workload params table row {index} is missing fields: {', '.join(missing)}")

    stage_raw = row["stage"]
    if isinstance(stage_raw, str):
        stage_key = stage_raw.strip().lower()
        if stage_key in _PROFILE_STAGE_VALUES:
            stage = _PROFILE_STAGE_VALUES[stage_key]
        else:
            stage = _parse_integer(stage_raw, "stage", index)
    else:
        stage = _parse_integer(stage_raw, "stage", index)
    if stage not in (1, 2):
        raise ValueError(
            f"workload params table row {index} field stage must be prefill/decode or 1/2")

    node_raw = row["node_type"]
    if isinstance(node_raw, str) and node_raw.strip().upper() in \
            _PROFILE_NODE_TYPE_VALUES:
        node_type = _PROFILE_NODE_TYPE_VALUES[node_raw.strip().upper()]
    else:
        node_type = _parse_integer(node_raw, "node_type", index)
    if node_type not in _PROFILE_NODE_TYPE_VALUES.values():
        raise ValueError(
            f"workload params table row {index} field node_type is unsupported")

    normalized = {
        "stage": stage,
        "node_type": node_type,
        "batch_size": _parse_integer(
            row["batch_size"], "batch_size", index),
        "sequence_tokens": _parse_integer(
            row["sequence_tokens"], "sequence_tokens", index),
        "duration_ns": _parse_integer(
            row["duration_ns"], "duration_ns", index),
        "comm_size_bytes": _parse_integer(
            row["comm_size_bytes"], "comm_size_bytes", index),
    }
    for field in ("batch_size", "sequence_tokens"):
        if normalized[field] != -1 and normalized[field] <= 0:
            raise ValueError(
                f"workload params table row {index} field {field} "
                "must be positive or -1")
    for field in ("duration_ns", "comm_size_bytes"):
        if normalized[field] < 0:
            raise ValueError(
                f"workload params table row {index} field {field} must be non-negative")
    return normalized


def _normalize_request(row, index):
    if not isinstance(row, dict):
        raise ValueError(f"request row {index} must be a dictionary")
    missing = [field for field in REQUEST_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"request row {index} is missing fields: {', '.join(missing)}")
    normalized = {
        field: _parse_integer(row[field], field, index)
        for field in REQUEST_FIELDS
    }
    for field, value in normalized.items():
        if value < 0:
            raise ValueError(
                f"request row {index} field {field} must be non-negative")
    if normalized["prompt_len"] == 0:
        raise ValueError(
            f"request row {index} field prompt_len must be positive")
    return normalized


def _parse_integer(value, field, index):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"request row {index} field {field} must be an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"request row {index} field {field} must be an integer") \
                from exc
    raise ValueError(
        f"request row {index} field {field} must be an integer")
