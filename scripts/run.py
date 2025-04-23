import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld
from parse_chakra import Node,Attribute,BoolList,Parse
from node_conversion import folder_to_int_array,test_conversion

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

# params
chakra_nodes_data_length=10000000 
chakra_nodes_num=1
# 

def empty_tensor(rows= chakra_nodes_num,max_len=chakra_nodes_data_length):
    tensor = torch.zeros((rows,max_len), dtype=torch.int32)
    return tensor

def ints_to_tensor(int_array,num_world=1,rows= chakra_nodes_num,max_len=chakra_nodes_data_length):
     i=0
     tensor = torch.zeros((num_world,rows,max_len), dtype=torch.int32)
     for array in int_array:
        tensor[0,i,:len(array)] = torch.tensor(array, dtype=torch.int32)
        i+=1
     return tensor
def tensor_to_ints(tensor):
     encoded = tensor.cpu().numpy().tolist() 
     return encoded
 
# def test_conversion():
#     json_file_path = '/home/zhangran/madrona2/2/madrona_simple_example/scripts/input/npu.0.json'
#     nodes = Parse(json_file_path)
    
#     if nodes:
#         first_node = nodes[1]
#         print(f"转换前的 Node 对象: {first_node}")
#         int_array = node_to_int_array(first_node)
#         print(f"转换后的整数数组: {int_array}")

#         node_back = int_array_to_node(int_array)
#         print(f"转换回的 Node 对象: {node_back}")
#  -----------------------------

# test_conversion()

folder_path = '/home/zhangran/madrona2/2/madrona_simple_example/scripts/input'
data=folder_to_int_array(folder_path)
data_tensor=ints_to_tensor(data)
int_tensor=tensor_to_ints(grid_world.chakra_nodes_data)
grid_world.chakra_nodes_data.copy_(data_tensor)
for i in range(5):
    # data_tensor=ints_to_tensor([[1],[2]])
   
    grid_world.step()
    
    print(f"Step {i+1} frame:\n")