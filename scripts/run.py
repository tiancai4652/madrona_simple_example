import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld


def string_to_tensor(s, max_len=1000):
    """
    将字符串编码为数值，并存储在张量中
    :param s: 要编码的字符串
    :param max_len: 张量的最大长度
    :return: 包含字符串编码的张量
    """
    # 将字符串转换为ASCII编码
    encoded = [ord(c) for c in s]
    # 创建张量并填充编码值
    tensor = torch.zeros(max_len, dtype=torch.int32)
    tensor[:len(encoded)] = torch.tensor(encoded, dtype=torch.int32)
    return tensor

def tensor_to_string(tensor):
    """
    将编码张量转换回字符串
    :param tensor: 包含字符串编码的张量
    :return: 解码后的字符串
    """
    encoded = tensor.cpu().numpy().tolist()  # 先将张量复制到 CPU
    s = ''.join(chr(int(i)) for i in encoded if int(i) != 0)  # 确保元素为整数
    return s


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

print(grid_world.observations.shape)

example_string = ""
encoded_string_tensor = string_to_tensor(example_string)
grid_world.results2.copy_(encoded_string_tensor)

for i in range(5):
    
    if i>0:
        results_tensor = grid_world.results2[0]
        decoded_string = tensor_to_string(results_tensor)
        example_string = decoded_string+" "+"hello"
        encoded_string_tensor = string_to_tensor(example_string)
        grid_world.results2.copy_(encoded_string_tensor)
    
    print("Obs:")
    print(grid_world.observations)

    # "Policy"
    grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))
    #grid_world.actions[:, 0] = 3 # right to win given (4, 4) start

    print("Actions:")
    print(grid_world.actions)

    # Advance simulation across all worlds
    grid_world.step()
    
    print("Rewards: ")
    print(grid_world.rewards)
    print("Dones:   ")
    print(grid_world.dones)
    
    print("Results:   ")
    print(grid_world.results)
    
    print("Results2:   ")
    # print(grid_world.results2)
    # 获取并解码字符串张量
    results_tensor = grid_world.results2[0]
    decoded_string = tensor_to_string(results_tensor)
    print(decoded_string)
    # print()
    
    print()
