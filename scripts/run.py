import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld

num_worlds = 1

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
chakra_nodes_data_length=10000 
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
#  -----------------------------

for i in range(5):
    data=empty_tensor()
    data[0]=1+i
    data[1]=2+i
    # int_tensor=ints_to_tensor(grid_world.chakra_nodes_data)
    grid_world.chakra_nodes_data.copy_(data)
    grid_world.step()