import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld

num_worlds = int(sys.argv[1])

enable_gpu_sim = False
if len(sys.argv) >= 3 and sys.argv[2] == '--gpu':
    enable_gpu_sim = True

array_shape = [5,6]
walls = np.zeros(array_shape)
rewards = np.zeros(array_shape)
walls[3, 2:] = 1
start_cell = np.array([4,5])
end_cell = np.array([[4,5]])
rewards[4, 0] = -1
rewards[4, 5] = 1

grid_world = GridWorld(num_worlds, start_cell, end_cell, rewards, walls, enable_gpu_sim, 0)

#grid_world.vis_world()


# --------------------------
chakra_nodes_data_length=47 * 1024 * 6000
def empty_tensor(max_len=chakra_nodes_data_length):
    tensor = torch.zeros(max_len, dtype=torch.int32)
    return tensor

def ints_to_tensor(int_array, max_len=chakra_nodes_data_length):
    encoded = int_array
    tensor = torch.zeros(max_len, dtype=torch.int32)
    tensor[:len(encoded)] = torch.tensor(encoded, dtype=torch.int32)
    return tensor
def tensor_to_ints(tensor):
    encoded = tensor.cpu().numpy().tolist() 
    return encoded


for i in range(5):
    
    data=empty_tensor()
    data[0]=1
    data[1]=2
    
    grid_world.step()

# print(grid_world.observations.shape)

# for i in range(5):
#     print("Obs:")
#     print(grid_world.observations)

#     # "Policy"
#     grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
#     #grid_world.actions[:, 0] = 3 # right to win given (4, 4) start

#     print("Actions:")
#     print(grid_world.actions)

#     # Advance simulation across all worlds
#     grid_world.step()
    
#     print("Rewards: ")
#     print(grid_world.rewards)
#     print("Dones:   ")
#     print(grid_world.dones)
#     print()
