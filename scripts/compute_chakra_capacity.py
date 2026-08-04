"""Compute the required CHAKRA_NODES_DATA_LENGTH for a Chakra workload dir.

Each Chakra node encodes to INTS_PER_NODE (=51) int32 values
(name 20 + type 1 + id 1 + dataDeps 10 + attr 6*3 + duration 1).

Usage:
  python compute_chakra_capacity.py <workload_dir_or_json> [--round-to N]
"""

import argparse
import json
import re
import sys
from pathlib import Path

INTS_PER_NODE = 51  # must match C++ sys_types.hpp INTS_PER_NODE


def count_nodes(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return 0
    return len(data)


def extract_number(filename):
    m = re.search(r"npu\.(\d+)\.json", Path(filename).name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(
        description="Compute max CHAKRA_NODES_DATA_LENGTH for a workload dir")
    parser.add_argument("path", help="workload dir (npu.<id>.json) or single json")
    parser.add_argument("--round-to", type=int, default=5000,
                        help="round the recommended length up to a multiple "
                             "of this (default 5000)")
    parser.add_argument("--raw", action="store_true",
                        help="print only the rounded required length and exit")
    parser.add_argument("--raw-nodes", action="store_true",
                        help="print only the max nodes per NPU and exit")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        files = sorted(path.rglob("*.json"), key=extract_number)
    elif path.is_file():
        files = [path]
    else:
        sys.exit(f"path does not exist: {path}")

    if not files:
        sys.exit(f"no json files found in {path}")

    counts = []
    per_file = []
    for f in files:
        try:
            n = count_nodes(f)
        except Exception as exc:
            print(f"warning: failed to parse {f}: {exc}", file=sys.stderr)
            continue
        counts.append(n)
        per_file.append((f.name, n))

    if not counts:
        sys.exit("no json files could be parsed")

    max_nodes = max(counts)
    max_file = per_file[counts.index(max_nodes)][0]
    max_ints = max_nodes * INTS_PER_NODE
    total_nodes = sum(counts)
    avg_nodes = total_nodes / len(counts)

    rounds = (max_ints + args.round_to - 1) // args.round_to * args.round_to

    if args.raw:
        print(rounds)
        return
    if args.raw_nodes:
        print(max_nodes)
        return

    print(f"json files           : {len(counts)}")
    print(f"total nodes          : {total_nodes}")
    print(f"nodes per NPU min    : {min(counts)}")
    print(f"nodes per NPU avg    : {avg_nodes:.1f}")
    print(f"nodes per NPU max    : {max_nodes}  ({max_file})")
    print(f"encoded ints per NPU : "
          f"{min(counts) * INTS_PER_NODE}..{max_ints}  ({INTS_PER_NODE} ints/node)")
    print(f"required CHAKRA_NODES_DATA_LENGTH = {max_ints}")
    print(f"recommended (round up to {args.round_to}) = {rounds}")
    if rounds > 50000:
        print("NOTE: exceeds the current default 50000; bump MAX_NPUS/"
              "CHAKRA_NODES_DATA_LENGTH in config.py and sys_types.hpp.")


if __name__ == "__main__":
    main()
