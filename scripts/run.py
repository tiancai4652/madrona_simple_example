import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld

import mmap
import posix_ipc
import struct
from input_helper import MadronaEvent,MadronaEvents,madrona_events_to_int_array,string_to_tensor,tensor_to_string,events_to_tensor,tensor_to_events,int_array_to_madrona_events

import comm

# read python params,set num_worlds and enable_gpu
if(len(sys.argv)==1):
    param1=1
    enable_gpu=False
else:
    param1=int(sys.argv[1])
    if len(sys.argv) >= 3 and sys.argv[2] == '--gpu':
        enable_gpu=True
num_worlds = param1
enable_gpu_sim =enable_gpu

# init 
array_shape = [5,6]
walls = np.zeros(array_shape)
rewards = np.zeros(array_shape)
walls[3, 2:] = 1
start_cell = np.array([4,5])
end_cell = np.array([[4,5]])
rewards[4, 0] = -1
rewards[4, 5] = 1
grid_world = GridWorld(num_worlds, start_cell, end_cell, rewards, walls, enable_gpu_sim, 0)


# memoryRW: set params string, for instance
example_string = ""
encoded_string_tensor = string_to_tensor(example_string)
grid_world.results2.copy_(encoded_string_tensor)

# to do
# memoryRW: read command and set field
def receive_set_command():
    # events = [
    #     MadronaEvent(22, 100, 200, 300, 400, 500),
    #     MadronaEvent(22, 101, 201, 301, 401, 501)
    # ]
    events=comm.receive_data()
    # madrona_events = MadronaEvents(events)
    # int_array = madrona_events_to_int_array(madrona_events)
    # int_tensor = events_to_tensor(int_array)
    # grid_world.madronaEvents.copy_(int_tensor)

    #  test read field
    # int_tensor=tensor_to_events(grid_world.madronaEvents)
    # m_events=int_array_to_madrona_events(int_tensor)
    
    m_events=events
    for event in m_events:
        event.print_event()
    # process
    for event in m_events:
        # firsr to process sim_get_time event, type is 3
        if event.type==3:
            time=grid_world.simulation_time.cpu().item()
            events = [MadronaEvent(event.type, event.eventId, time, 0, 0, 0)]
            send_command(events)
            receive_set_command()
        # second, we process sim_schedule and sim_send, type is 2,0
        if event.type==2 or event.type==0:
            # read event buffer
            int_tensor=tensor_to_events(grid_world.madronaEvents)
            m_events=int_array_to_madrona_events(int_tensor)
            # add new event
            m_events.events.append(event)
            # set back
            madrona_events = MadronaEvents(m_events)
            int_array = madrona_events_to_int_array(madrona_events)
            int_tensor = events_to_tensor(int_array)
            grid_world.madronaEvents.copy_(int_tensor)
            
            int_tensor=tensor_to_events(grid_world.madronaEvents)
            m_events=int_array_to_madrona_events(int_tensor)
            for event in m_events.events:
                event.print_event()
            
            grid_world.step()
            
            
            

def send_command(events):
    comm.send_data(events)
    
for i in range(5):
# while True:    
    print("cpu:")
    
    print("time:")
    print(grid_world.simulation_time.cpu().item())
    # read command, set field
    receive_set_command()
    
    # events = [
    #     MadronaEvent(22, 100, 200, 300, 400, 500),
    #     MadronaEvent(22, 101, 201, 301, 401, 501)
    # ]
    # madrona_events = MadronaEvents(events)
    # int_array = madrona_events_to_int_array(madrona_events)
    # int_tensor = events_to_tensor(int_array)
    # grid_world.madronaEvents.copy_(int_tensor)
    # int_tensor=tensor_to_events(grid_world.madronaEvents)
    # m_events=int_array_to_madrona_events(int_tensor)
    # for event in m_events.events:
    #     event.print_event()
    
    # process next frame.
    # grid_world.step()
    
    # to do
    # read command result filed and send command.
    # send_command()
    
    # to do
    # check if end
    # quit()
    
    print()
