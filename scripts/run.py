import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld
from parse_chakra import Node,Attribute,BoolList,Parse
from node_conversion import node_to_int_array,int_array_to_node

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
 
def test_conversion():
    json_file_path = '/home/zhangran/madrona2/2/madrona_simple_example/scripts/input/npu.0.json'
    nodes = Parse(json_file_path)
    
    if nodes:
        first_node = nodes[1]
        print(f"转换前的 Node 对象: {first_node}")
        int_array = node_to_int_array(first_node)
        print(f"转换后的整数数组: {int_array}")

        node_back = int_array_to_node(int_array)
        print(f"转换回的 Node 对象: {node_back}")
#  -----------------------------

for i in range(5):
    data=empty_tensor()
    data[0]=1+i
    data[1]=2+i
    int_tensor=tensor_to_ints(grid_world.chakra_nodes_data)
    grid_world.chakra_nodes_data.copy_(data)
    grid_world.step()