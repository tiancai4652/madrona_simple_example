import csv
import numpy as np
from ._madrona_simple_example_cpp import SimpleGridworldSimulator, madrona

__all__ = [
    'GridWorld',
    'make_default_network_inputs',
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


def load_network_inputs_from_files(topology_path, flow_path):
    with open(topology_path, 'r', encoding='utf-8') as f:
        topo_inputs = _parse_topology_lines(f)
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
            ):
        self.size = np.array(walls.shape)
        self.start_cell = start_cell
        self.end_cells = end_cells
        self.rewards_input = rewards
        self.walls = walls
        self.network_inputs = make_default_network_inputs() if network_inputs is None else network_inputs

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

    def step(self):
        self.sim.step()

    def simulation_time(self):
        return self.sim.simulation_time()

    def num_flow_defs(self):
        return self.sim.num_flow_defs()

    def num_pending_flows(self):
        return self.sim.num_pending_flows()

    def num_delayed_events(self):
        return self.sim.num_delayed_events()

    def num_active_tags(self):
        return self.sim.num_active_tags()

    def num_source_tags(self):
        return self.sim.num_source_tags()

    def flow_completions(self):
        return [
            self.sim.flow_completion(i)
            for i in range(self.sim.num_flow_completions())
        ]

    def write_flow_completion_csv(self, output_path):
        rows = sorted(self.flow_completions(), key=lambda row: row['flow_id'])
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
