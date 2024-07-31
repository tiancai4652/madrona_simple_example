import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld

import mmap
import posix_ipc
import struct
from input_helper import MadronaEvent,MadronaEvents,madrona_events_to_int_array,string_to_tensor,tensor_to_string,events_to_tensor,tensor_to_events,int_array_to_madrona_events


def comm():
    memory = posix_ipc.SharedMemory("myshm")
    map_file = mmap.mmap(memory.fd, mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
    semaphore_a = posix_ipc.Semaphore("semA")
    semaphore_b = posix_ipc.Semaphore("semB")

    # 等待 C++ 程序的数据
    semaphore_a.acquire()
    data = map_file.read(4)
    value = struct.unpack('i', data)[0]
    print(f"Python: Received from C++: {value}")

    # 处理数据并写回
    new_value = value + 100
    map_file.seek(0)
    map_file.write(struct.pack('i', new_value))
    print(f"Python: Data processed and written back: {new_value}")

    # 通知 C++ 程序处理完成
    semaphore_b.release()

    # 清理资源
    map_file.close()
    memory.close_fd()
    semaphore_a.close()
    semaphore_b.close()


# num_worlds = int(sys.argv[1])
num_worlds = 1

enable_gpu_sim = False
# if len(sys.argv) >= 3 and sys.argv[2] == '--gpu':
#     enable_gpu_sim = True

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

# events = [
#   22, 100, 200, 300, 400, 500
# ]

events = [
    MadronaEvent(22, 100, 200, 300, 400, 500),
    MadronaEvent(22, 101, 201, 301, 401, 501)
]
madrona_events = MadronaEvents(events)
int_array = madrona_events_to_int_array(madrona_events)

int_tensor = events_to_tensor(int_array)
grid_world.madronaEvents.copy_(int_tensor)
int_tensor=tensor_to_events(grid_world.madronaEvents)
m_events=int_array_to_madrona_events(int_tensor)
print(m_events.events[0].time)

for i in range(5):
# i=0
# while True:
    i=i+1 
    if i>0:
        # results_tensor = grid_world.results2[0]
        # decoded_string = tensor_to_string(results_tensor)
        # example_string = decoded_string+" "+"hello"
        # encoded_string_tensor = string_to_tensor(example_string)
        encoded_string_tensor=string_to_tensor("hello")
        grid_world.results2.copy_(encoded_string_tensor)
    
    print("grid_world.madronaEvents:")
    int_tensor=tensor_to_events(grid_world.madronaEvents)
    m_events=int_array_to_madrona_events(int_tensor)
    print(m_events.events[0].time)
    
    # print("Obs:")
    # print(grid_world.observations)

    # grid_world.actions[:, 0] = torch.randint(0, 4, size=(num_worlds,))

    # print("Actions:")
    # print(grid_world.actions)

    grid_world.step()
    
    # print("Rewards: ")
    # print(grid_world.rewards)
    # print("Dones:   ")
    # print(grid_world.dones)
    
    # print("Results:   ")
    # print(grid_world.results)
    
    # print("Results2:   ")
    # results_tensor = grid_world.results2[0]
    # decoded_string = tensor_to_string(results_tensor)
    # print(decoded_string)

    
    print()
