import csv
import numpy as np
import torch
from ._madrona_simple_example_cpp import SimpleGridworldSimulator, madrona
from .chakra import build_process_params, load_chakra_workload
from .inference import (
    INFERENCE_STATS_OUTPUT_FIELDS,
    load_request_trace,
    load_workload_params_table,
    pack_request_trace,
    pack_workload_params_table,
    parse_inference_stats,
)

__all__ = [
    'GridWorld',
    'make_default_network_inputs',
    'make_empty_flow_inputs',
    'load_network_inputs_from_files',
]


def _parse_topology_lines(lines):
    node_ids = []
    node_types = []
    node_port_bws = []
    link_srcs = []
    link_dsts = []
    link_delays = []
    link_bandwidths = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        cmd = parts[0].upper()
        if cmd == 'NODE':
            if len(parts) < 3:
                raise ValueError(f'invalid NODE line: {raw_line.rstrip()}')
            node_type_str = parts[1].upper()
            node_id = int(parts[2])
            port_bw = float(parts[3]) if len(parts) >= 4 else 100.0
            if node_type_str in ('H', 'HOST'):
                node_type = 0
            elif node_type_str in ('S', 'SWITCH'):
                node_type = 1
            else:
                raise ValueError(f'unknown node type: {parts[1]}')

            node_ids.append(node_id)
            node_types.append(node_type)
            node_port_bws.append(port_bw)
        elif cmd == 'LINK':
            if len(parts) < 4:
                raise ValueError(f'invalid LINK line: {raw_line.rstrip()}')
            link_srcs.append(int(parts[1]))
            link_dsts.append(int(parts[2]))
            link_delays.append(float(parts[3]))
            link_bandwidths.append(float(parts[4]) if len(parts) >= 5 else 0.0)
        else:
            raise ValueError(f'unknown topology command: {parts[0]}')

    return {
        'node_ids': np.array(node_ids, dtype=np.int32),
        'node_types': np.array(node_types, dtype=np.int32),
        'node_port_bws': np.array(node_port_bws, dtype=np.float64),
        'link_srcs': np.array(link_srcs, dtype=np.int32),
        'link_dsts': np.array(link_dsts, dtype=np.int32),
        'link_delays': np.array(link_delays, dtype=np.float64),
        'link_bandwidths': np.array(link_bandwidths, dtype=np.float64),
    }


def _parse_flow_lines(lines):
    flow_ids = []
    flow_src_nodes = []
    flow_dst_nodes = []
    flow_sizes = []
    flow_start_times = []
    flow_priorities = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        cmd = parts[0].upper()
        if cmd != 'FLOW':
            raise ValueError(f'unknown flow command: {parts[0]}')
        if len(parts) < 6:
            raise ValueError(f'invalid FLOW line: {raw_line.rstrip()}')

        flow_ids.append(int(parts[1]))
        flow_src_nodes.append(int(parts[2]))
        flow_dst_nodes.append(int(parts[3]))
        flow_sizes.append(float(parts[4]))
        flow_start_times.append(float(parts[5]))
        flow_priorities.append(int(parts[6]) if len(parts) >= 7 else 0)

    return {
        'flow_ids': np.array(flow_ids, dtype=np.int64),
        'flow_src_nodes': np.array(flow_src_nodes, dtype=np.int32),
        'flow_dst_nodes': np.array(flow_dst_nodes, dtype=np.int32),
        'flow_sizes': np.array(flow_sizes, dtype=np.float64),
        'flow_start_times': np.array(flow_start_times, dtype=np.float64),
        'flow_priorities': np.array(flow_priorities, dtype=np.int32),
    }


def make_empty_flow_inputs():
    return {
        'flow_ids': np.array([], dtype=np.int64),
        'flow_src_nodes': np.array([], dtype=np.int32),
        'flow_dst_nodes': np.array([], dtype=np.int32),
        'flow_sizes': np.array([], dtype=np.float64),
        'flow_start_times': np.array([], dtype=np.float64),
        'flow_priorities': np.array([], dtype=np.int32),
    }


def load_network_inputs_from_files(topology_path, flow_path=None):
    with open(topology_path, 'r', encoding='utf-8') as f:
        topo_inputs = _parse_topology_lines(f)
    if flow_path is None:
        flow_inputs = make_empty_flow_inputs()
    else:
        with open(flow_path, 'r', encoding='utf-8') as f:
            flow_inputs = _parse_flow_lines(f)

    merged = dict(topo_inputs)
    merged.update(flow_inputs)
    return merged


def make_default_network_inputs():
    host_bw = 25000000.0
    fabric_bw = 800000000.0
    host_leaf_delay = 0.25
    leaf_spine_delay = 0.5

    node_ids = np.arange(67, dtype=np.int32)
    node_types = np.zeros(67, dtype=np.int32)
    node_types[64:] = 1
    node_port_bws = np.full(67, host_bw, dtype=np.float64)
    node_port_bws[64:] = fabric_bw

    link_srcs = []
    link_dsts = []
    link_delays = []
    link_bandwidths = []

    for host_id in range(32):
        link_srcs.append(host_id)
        link_dsts.append(64)
        link_delays.append(host_leaf_delay)
        link_bandwidths.append(host_bw)

    for host_id in range(32, 64):
        link_srcs.append(host_id)
        link_dsts.append(65)
        link_delays.append(host_leaf_delay)
        link_bandwidths.append(host_bw)

    link_srcs.extend([64, 65])
    link_dsts.extend([66, 66])
    link_delays.extend([leaf_spine_delay, leaf_spine_delay])
    link_bandwidths.extend([fabric_bw, fabric_bw])

    flow_ids = []
    flow_src_nodes = []
    flow_dst_nodes = []
    flow_sizes = []
    flow_start_times = []
    flow_priorities = []

    flow_size = 2097152.0
    start_time = 0.002
    domain_size = 8
    num_domains = 8
    flow_id = 1
    for domain in range(num_domains):
        domain_base = domain * domain_size
        for src in range(domain_base, domain_base + domain_size):
            for dst in range(domain_base, domain_base + domain_size):
                if src == dst:
                    continue
                flow_ids.append(flow_id)
                flow_src_nodes.append(src)
                flow_dst_nodes.append(dst)
                flow_sizes.append(flow_size)
                flow_start_times.append(start_time)
                flow_priorities.append(0)
                flow_id += 1

    return {
        'node_ids': np.array(node_ids, dtype=np.int32),
        'node_types': np.array(node_types, dtype=np.int32),
        'node_port_bws': np.array(node_port_bws, dtype=np.float64),
        'link_srcs': np.array(link_srcs, dtype=np.int32),
        'link_dsts': np.array(link_dsts, dtype=np.int32),
        'link_delays': np.array(link_delays, dtype=np.float64),
        'link_bandwidths': np.array(link_bandwidths, dtype=np.float64),
        'flow_ids': np.array(flow_ids, dtype=np.int64),
        'flow_src_nodes': np.array(flow_src_nodes, dtype=np.int32),
        'flow_dst_nodes': np.array(flow_dst_nodes, dtype=np.int32),
        'flow_sizes': np.array(flow_sizes, dtype=np.float64),
        'flow_start_times': np.array(flow_start_times, dtype=np.float64),
        'flow_priorities': np.array(flow_priorities, dtype=np.int32),
    }


def _as_per_world(value_w0, value_w1, num_worlds, name,
                   list_values=None):
    """Broadcast-or-split helper for per-world inputs (004-多世界 P0,
    011-十世界验证 N-world extension).

    Single-world callers keep passing one value (old behavior, broadcast
    when num_worlds == 1). Two-world callers optionally pass a distinct
    second value via the ``*_w1`` parameter; otherwise world1 reuses the
    world0 value. N-world (3..10) callers pass ``list_values`` (a list
    with num_worlds entries; None entries broadcast the world0 value).
    ``*_w1`` and ``list_values`` are mutually exclusive; ``*_w1`` with
    num_worlds != 2 keeps the P0 error behavior.
    """
    if list_values is not None:
        if value_w1 is not None:
            raise ValueError(
                f"{name}_w1 and {name}s are mutually exclusive")
        if len(list_values) != num_worlds:
            raise ValueError(
                f"{name}s length {len(list_values)} != "
                f"num_worlds={num_worlds}")
        return [value_w0 if v is None else v for v in list_values]
    if num_worlds == 1:
        if value_w1 is not None:
            raise ValueError(f"{name}_w1 requires num_worlds=2")
        return [value_w0]
    if num_worlds == 2:
        if value_w1 is None:
            return [value_w0, value_w0]
        return [value_w0, value_w1]
    # N > 2 without a list: broadcast world0 (same base).
    if value_w1 is not None:
        raise ValueError(
            f"{name}_w1 only supports num_worlds=2; "
            f"use {name}s for num_worlds={num_worlds}")
    return [value_w0] * num_worlds


class GridWorld:
    def __init__(self,
                 num_worlds,
                 start_cell,
                 end_cells,
                 rewards,
                 walls,
                 gpu_sim = False,
                 gpu_id = 0,
                 network_inputs = None,
                 propagation_interval = 0.0,
                 enable_pfc = 0,
                 pfc_egress = 0,
                 pfc_xoff_threshold = 1e9,
                 pfc_xon_threshold = 0.5e9,
                 dt_min = 0.0,
                 qos_mode = 0,
                 prior_weights = None,
                 system_workload = None,
                 system_config = None,
                 inference_config = None,
                 request_trace = None,
                 workload_params_table = None,
                  # 004-多世界 P0: optional per-world overrides for world1.
                  # When None they broadcast the world0 value (old behavior).
                  system_workload_w1 = None,
                  system_config_w1 = None,
                  inference_config_w1 = None,
                  request_trace_w1 = None,
                  workload_params_table_w1 = None,
                  # 011-十世界验证: optional per-world lists (length ==
                  # num_worlds; None entries broadcast the world0 value).
                  # Mutually exclusive with the ``*_w1`` pair above.
                  system_workloads = None,
                  system_configs = None,
                  inference_configs = None,
                  request_traces = None,
                  workload_params_tables = None,
            ):
        self.size = np.array(walls.shape)
        self.start_cell = start_cell
        self.end_cells = end_cells
        self.rewards_input = rewards
        self.walls = walls
        self.network_inputs = make_default_network_inputs() if network_inputs is None else network_inputs

        if (system_workload is None) != (system_config is None):
            raise ValueError(
                "system_workload and system_config must be provided together")
        if (inference_config is None) != (request_trace is None):
            raise ValueError(
                "inference_config and request_trace must be provided together")
        if inference_config is not None and system_config is None:
            raise ValueError(
                "inference_config requires system_workload and system_config")
        self.num_worlds = num_worlds
        if not 1 <= num_worlds <= 10:
            raise ValueError(
                f"num_worlds={num_worlds} is not supported "
                f"(011 supports 1..10)")

        system_workloads = _as_per_world(
            system_workload, system_workload_w1, num_worlds,
            "system_workload", system_workloads)
        system_configs = _as_per_world(
            system_config, system_config_w1, num_worlds, "system_config",
            system_configs)
        inference_configs = _as_per_world(
            inference_config, inference_config_w1, num_worlds,
            "inference_config", inference_configs)
        request_traces = _as_per_world(
            request_trace, request_trace_w1, num_worlds, "request_trace",
            request_traces)
        workload_params_tables = _as_per_world(
            workload_params_table, workload_params_table_w1, num_worlds,
            "workload_params_table", workload_params_tables)

        for world_idx, cfg in enumerate(system_configs):
            if cfg is not None:
                cfg.validate()
                node_ids = np.asarray(self.network_inputs['node_ids'])
                node_types = np.asarray(self.network_inputs['node_types'])
                host_ids = set(node_ids[node_types == 0].astype(int).tolist())
                missing = [npu_id for npu_id in range(cfg.npu_count)
                           if npu_id not in host_ids]
                if missing:
                    raise ValueError(
                        f"world{world_idx}: network topology is missing NPU "
                        f"host ids: {missing[:8]}")

        # Per-world parsed inference inputs (None entry when disabled).
        # Row counts may differ per world (变长/变大小兼容: different
        # lengths/sizes/counts/intervals are row values, not shapes).
        inference_traces = [None] * num_worlds
        inference_profiles = [None] * num_worlds
        for world_idx in range(num_worlds):
            if inference_configs[world_idx] is not None:
                inference_configs[world_idx].validate(
                    system_configs[world_idx].npu_count
                    if system_configs[world_idx] is not None else None)
                inference_traces[world_idx] = load_request_trace(
                    request_traces[world_idx])
                inference_profiles[world_idx] = load_workload_params_table(
                    workload_params_tables[world_idx])

        if prior_weights is None:
            prior_weights = np.zeros(8, dtype=np.float64)
        else:
            prior_weights = np.array(prior_weights).astype(np.float64)
            if len(prior_weights) < 8:
                prior_weights = np.pad(prior_weights, (0, 8 - len(prior_weights)))

        self.sim = SimpleGridworldSimulator(
                walls = np.array(walls).astype(np.bool_),
                rewards = np.array(rewards).astype(np.float32),
                end_cells = np.array(end_cells).astype(np.int32),
                start_x = start_cell[1],
                start_y = start_cell[0],
                node_ids = np.array(self.network_inputs['node_ids']).astype(np.int32),
                node_types = np.array(self.network_inputs['node_types']).astype(np.int32),
                node_port_bws = np.array(self.network_inputs['node_port_bws']).astype(np.float64),
                link_srcs = np.array(self.network_inputs['link_srcs']).astype(np.int32),
                link_dsts = np.array(self.network_inputs['link_dsts']).astype(np.int32),
                link_delays = np.array(self.network_inputs['link_delays']).astype(np.float64),
                link_bandwidths = np.array(self.network_inputs['link_bandwidths']).astype(np.float64),
                flow_ids = np.array(self.network_inputs['flow_ids']).astype(np.int64),
                flow_src_nodes = np.array(self.network_inputs['flow_src_nodes']).astype(np.int32),
                flow_dst_nodes = np.array(self.network_inputs['flow_dst_nodes']).astype(np.int32),
                flow_sizes = np.array(self.network_inputs['flow_sizes']).astype(np.float64),
                flow_start_times = np.array(self.network_inputs['flow_start_times']).astype(np.float64),
                flow_priorities = np.array(self.network_inputs['flow_priorities']).astype(np.int32),
                max_episode_length = 0,
                exec_mode = madrona.ExecMode.CUDA if gpu_sim else madrona.ExecMode.CPU,
                num_worlds = num_worlds,
                gpu_id = gpu_id,
                propagation_interval = propagation_interval,
                enable_pfc = enable_pfc,
                pfc_egress = pfc_egress,
                pfc_xoff_threshold = pfc_xoff_threshold,
                pfc_xon_threshold = pfc_xon_threshold,
                dt_min = dt_min,
                qos_mode = qos_mode,
                prior_weights = prior_weights,
            )

        self.force_reset = self.sim.reset_tensor().to_torch()
        self.actions = self.sim.action_tensor().to_torch()
        self.observations = self.sim.observation_tensor().to_torch()
        self.rewards = self.sim.reward_tensor().to_torch()
        self.dones = self.sim.done_tensor().to_torch()

        self._system_enabled = system_workloads[0] is not None
        self._inference_enabled = inference_configs[0] is not None
        # Per-world request counts (backward compat: single int when
        # num_worlds == 1, list when 2). Use num_inference_requests().
        self._num_inference_requests = [
            len(trace) if trace is not None else 0
            for trace in inference_traces
        ]
        if self._system_enabled:
            chakra_tensor = self.sim.chakra_nodes_data_tensor().to_torch()
            params_tensor = self.sim.process_params_tensor().to_torch()
            for world_idx in range(num_worlds):
                chakra_data = load_chakra_workload(
                    system_workloads[world_idx],
                    system_configs[world_idx].npu_count)
                process_params = build_process_params(
                    system_configs[world_idx])
                chakra_tensor[world_idx].copy_(
                    torch.from_numpy(chakra_data))
                params_tensor[world_idx].copy_(
                    torch.from_numpy(process_params))

        if self._inference_enabled:
            config_tensor = self.sim.inference_config_tensor().to_torch()
            request_tensor = self.sim.inference_request_tensor().to_torch()
            profile_tensor = self.sim.workload_params_table_tensor().to_torch()
            for world_idx in range(num_worlds):
                config_row = torch.from_numpy(
                    inference_configs[world_idx].pack(
                        self._num_inference_requests[world_idx],
                        system_configs[world_idx].npu_count,
                        len(inference_profiles[world_idx])))
                request_row = torch.from_numpy(pack_request_trace(
                    inference_traces[world_idx]))
                profile_row = torch.from_numpy(pack_workload_params_table(
                    inference_profiles[world_idx]))
                config_tensor[world_idx].copy_(config_row)
                request_tensor[world_idx].copy_(request_row)
                profile_tensor[world_idx].copy_(profile_row)

    def step(self):
        self.sim.step()

    def _check_world(self, world_idx):
        if not 0 <= world_idx < self.num_worlds:
            raise ValueError(
                f"world_idx={world_idx} out of range "
                f"(num_worlds={self.num_worlds})")
        return world_idx

    def num_inference_requests(self, world_idx=0):
        """Per-world request count (list form stored internally)."""
        self._check_world(world_idx)
        counts = self._num_inference_requests
        if isinstance(counts, list):
            return counts[world_idx]
        return counts

    def simulation_time(self, world_idx=0):
        return self.sim.simulation_time(self._check_world(world_idx))

    def system_status(self, world_idx=0):
        raw = self.sim.system_status_tensor().to_torch()[
            self._check_world(world_idx)].cpu().tolist()
        return {
            'initialized': bool(raw[0]),
            'finished': bool(raw[1]),
            'failed': bool(raw[2]),
            'error_code': int(raw[3]),
            'finished_npus': int(raw[4]),
            'total_npus': int(raw[5]),
        }

    def system_finished(self, world_idx=0):
        return self.system_status(world_idx)['finished']

    def system_failed(self, world_idx=0):
        return self.system_status(world_idx)['failed']

    def all_worlds_finished(self):
        return all(self.system_finished(w) for w in range(self.num_worlds))

    def num_flow_defs(self, world_idx=0):
        return self.sim.num_flow_defs(self._check_world(world_idx))

    def num_pending_flows(self, world_idx=0):
        return self.sim.num_pending_flows(self._check_world(world_idx))

    def num_delayed_events(self, world_idx=0):
        return self.sim.num_delayed_events(self._check_world(world_idx))

    def num_active_tags(self, world_idx=0):
        return self.sim.num_active_tags(self._check_world(world_idx))

    def num_source_tags(self, world_idx=0):
        return self.sim.num_source_tags(self._check_world(world_idx))

    def num_flow_completions(self, world_idx=0):
        return self.sim.num_flow_completions(self._check_world(world_idx))

    def last_step_phase_times(self, world_idx=0):
        return dict(self.sim.last_step_phase_times(
            self._check_world(world_idx)))

    def flow_completions(self, world_idx=0):
        self._check_world(world_idx)
        return [
            self.sim.flow_completion(i, world_idx)
            for i in range(self.sim.num_flow_completions(world_idx))
        ]

    def inference_stats(self, world_idx=0):
        if not self._inference_enabled:
            return []
        self._check_world(world_idx)
        raw = self.sim.inference_stats_tensor().to_torch()[world_idx].cpu().numpy()
        return parse_inference_stats(raw, self.num_inference_requests(world_idx))

    def write_inference_stats_csv(self, output_path, world_idx=0):
        # world_idx=None + a per-world path list writes one CSV per
        # world (multi output for multi-world runs).
        if world_idx is None:
            paths = list(output_path)
            if len(paths) != self.num_worlds:
                raise ValueError(
                    "multi output requires one path per world, got "
                    f"{len(paths)} for num_worlds={self.num_worlds}")
            for world, path in enumerate(paths):
                self.write_inference_stats_csv(path, world)
            return
        rows = self.inference_stats(world_idx)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=INFERENCE_STATS_OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def write_flow_completion_csv(self, output_path, world_idx=0):
        if world_idx is None:
            paths = list(output_path)
            if len(paths) != self.num_worlds:
                raise ValueError(
                    "multi output requires one path per world, got "
                    f"{len(paths)} for num_worlds={self.num_worlds}")
            for world, path in enumerate(paths):
                self.write_flow_completion_csv(path, world)
            return
        rows = sorted(self.flow_completions(world_idx), key=lambda row: row['flow_id'])
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'flow_id',
                'src_node',
                'dst_node',
                'size_bytes',
                'start_time_ms',
                'end_time_ms',
                'fct_ms',
                'priority',
            ])
            for row in rows:
                writer.writerow([
                    row['flow_id'],
                    row['src_node'],
                    row['dst_node'],
                    f"{row['size_bytes']:.6f}",
                    f"{row['start_time_ms']:.6f}",
                    f"{row['end_time_ms']:.6f}",
                    f"{row['fct_ms']:.6f}",
                    row['priority'],
                ])
