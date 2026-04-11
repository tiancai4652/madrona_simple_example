import sys
from pathlib import Path

import numpy as np
import torch
from madrona_simple_example import GridWorld, load_network_inputs_from_files


def load_network_inputs_from_args(args):
    if len(args) >= 4:
        topo_path = Path(args[2])
        flow_path = Path(args[3])
        return load_network_inputs_from_files(topo_path, flow_path)
    return None


num_worlds = int(sys.argv[1])

enable_gpu_sim = False
arg_offset = 2
if len(sys.argv) >= 3 and sys.argv[2] == '--gpu':
    enable_gpu_sim = True
    arg_offset = 3

network_inputs = None
if len(sys.argv) > arg_offset + 1:
    network_inputs = load_network_inputs_from_args([
        sys.argv[0],
        sys.argv[1],
        sys.argv[arg_offset],
        sys.argv[arg_offset + 1],
    ])

array_shape = [5, 6]
walls = np.zeros(array_shape)
rewards = np.zeros(array_shape)
walls[3, 2:] = 1
start_cell = np.array([4, 5])
end_cell = np.array([[4, 5]])
rewards[4, 0] = -1
rewards[4, 5] = 1

grid_world = GridWorld(
    num_worlds,
    start_cell,
    end_cell,
    rewards,
    walls,
    enable_gpu_sim,
    0,
    network_inputs=network_inputs,
)

print(grid_world.observations.shape)

for i in range(5):
    print("Obs:")
    print(grid_world.observations)

    grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))

    print("Actions:")
    print(grid_world.actions)

    grid_world.step()

    print("Rewards: ")
    print(grid_world.rewards)
    print("Dones:   ")
    print(grid_world.dones)
    print()
