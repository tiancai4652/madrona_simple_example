"""Compute required capacity / slot parameters for a Chakra workload dir.

Each Chakra node encodes to INTS_PER_NODE (=51) int32 values
(name 20 + type 1 + id 1 + dataDeps 10 + attr 6*3 + duration 1).

Covers the workload-dependent "卡槽" parameters of the merged flow
simulator:

  NPU_NUM                   = number of npu.*.json files
  CHAKRA_NODES_DATA_LENGTH  = max nodes/NPU * 51 (rounded)
  MAX_CHAKRA_NODES_PER_NPU  = max nodes/NPU
  MAX_FLOWS_PER_NPU         = max SEND/NPU (safe in-flight cap)
  MAX_COMM_TASK_PER_NPU     = max(SEND, RECV)/NPU
  MAX_FLOW_PAIR_STATES      = total SEND flows (persistent pair-state table)
  ring dims                 = factors whose product == NPU_NUM

Usage:
  python compute_chakra_capacity.py <workload_dir_or_json> [--round-to N]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

INTS_PER_NODE = 51  # must match C++ sys_types.hpp INTS_PER_NODE


def count_nodes(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return 0
    return len(data)


def node_type_counts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return Counter()
    return Counter(n.get("type") for n in data)


def count_flows(path):
    """Count SEND / RECV nodes and distinct comm_para (flow id) values."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return 0, 0, set()
    sends = 0
    recvs = 0
    paras = set()
    for n in data:
        t = n.get("type")
        if t == "COMM_SEND_NODE":
            sends += 1
        elif t == "COMM_RECV_NODE":
            recvs += 1
        for a in n.get("attr", []):
            if a.get("name") == "comm_para":
                v = a.get("uint32Val") or a.get("int64Val") or a.get("uint64Val")
                if v is not None:
                    paras.add(int(v))
    return sends, recvs, paras


def extract_number(filename):
    m = re.search(r"npu\.(\d+)\.json", Path(filename).name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(
        description="Compute required capacity / slot parameters for a workload")
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
    type_counters = []
    flow_stats = []
    for f in files:
        try:
            n = count_nodes(f)
            tc = node_type_counts(f)
            sends, recvs, paras = count_flows(f)
        except Exception as exc:
            print(f"warning: failed to parse {f}: {exc}", file=sys.stderr)
            continue
        counts.append(n)
        per_file.append((f.name, n))
        type_counters.append(tc)
        flow_stats.append((sends, recvs, paras))

    if not counts:
        sys.exit("no json files could be parsed")

    max_nodes = max(counts)
    max_file = per_file[counts.index(max_nodes)][0]
    max_ints = max_nodes * INTS_PER_NODE
    total_nodes = sum(counts)
    avg_nodes = total_nodes / len(counts)
    npu_count = len(counts)

    # Flow-level aggregates.
    max_sends = max(s for s, _, _ in flow_stats)
    max_recvs = max(r for _, r, _ in flow_stats)
    total_sends = sum(s for s, _, _ in flow_stats)
    total_recvs = sum(r for _, r, _ in flow_stats)
    all_paras = set()
    for _, _, paras in flow_stats:
        all_paras |= paras
    with_para_sends = sum(1 for s, _, paras in flow_stats if s and paras)

    rounds = (max_ints + args.round_to - 1) // args.round_to * args.round_to

    if args.raw:
        print(rounds)
        return
    if args.raw_nodes:
        print(max_nodes)
        return

    print(f"json files           : {npu_count}")
    print(f"total nodes          : {total_nodes}")
    print(f"nodes per NPU max    : {max_nodes}  ({max_file})")
    print(f"encoded ints per NPU : {max_ints}  ({INTS_PER_NODE} ints/node)")
    print()
    print("--- 卡槽参数（按此配置 sys_types.hpp / types.hpp）---")
    print(f"NPU_NUM                  = {npu_count}")
    print(f"CHAKRA_NODES_DATA_LENGTH = {rounds}   (max_nodes x {INTS_PER_NODE}, round to {args.round_to})")
    print(f"MAX_CHAKRA_NODES_PER_NPU = {max_nodes}")
    print(f"MAX_FLOWS_PER_NPU        = {max_sends}   (max SEND/NPU, 在飞流安全上限)")
    print(f"MAX_COMM_TASK_PER_NPU    = {max(max_sends, max_recvs)}   (max(SEND,RECV)/NPU)")
    print(f"MAX_FLOW_PAIR_STATES     = {total_sends}   (总 SEND 流数, 持久配对表)")
    print(f"ring_dims                = 乘积 == {npu_count} 的因子组合, 如 ({npu_count},1,1)")
    print()
    print(f"SEND 流总数 : {total_sends}")
    print(f"RECV 流总数 : {total_recvs}")
    print(f"不同 comm_para(flow_id) 数: {len(all_paras)}")
    if with_para_sends > 0:
        print("注: 节点带 comm_para, RECV 配对走持久状态表 (flow_id 路径)")
    else:
        print("注: 节点无 comm_para, RECV 配对走邮箱 (src,dst) 回退路径")
    if len(all_paras) > 0 and len(all_paras) < total_sends:
        print("警告: comm_para 不是全局唯一, 状态表按 (comm_para,src,dst) 复合键消歧")


if __name__ == "__main__":
    main()