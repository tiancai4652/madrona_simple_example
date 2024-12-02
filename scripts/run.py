import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld

import mmap
import posix_ipc
import struct
from input_helper import MadronaEvent,MadronaEvents,empty_tensor,madrona_events_to_int_array,string_to_tensor,tensor_to_string,events_to_tensor,tensor_to_events,int_array_to_madrona_events

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
gpu_id=0

# init 
array_shape = [5,6]
walls = np.zeros(array_shape)
rewards = np.zeros(array_shape)
walls[3, 2:] = 1
start_cell = np.array([4,5])
end_cell = np.array([[4,5]])
rewards[4, 0] = -1
rewards[4, 5] = 1
grid_world = GridWorld(num_worlds, start_cell, end_cell, rewards, walls, enable_gpu_sim, gpu_id)


flow_events=[]
flow_events_result=[]

time_current=None

# memoryRW: set params string, for instance
example_string = ""
encoded_string_tensor = string_to_tensor(example_string,1000)
grid_world.results2.copy_(encoded_string_tensor)


def send_command(events):
    comm.send_data(events)
    
def check_result_reponse():
    int_tensor=tensor_to_events(grid_world.madronaEventsResult)
    m_events=int_array_to_madrona_events(int_tensor)
    if len(m_events.events)>0 or len(flow_events_result)>0:
        for event in m_events.events:
            flow_events_result.append(event)
        grid_world.madronaEventsResult.copy_(empty_tensor())
        if len(flow_events_result)>0:
            first_element = flow_events_result.pop(0)
            send_command([first_element])
            return
            # receive_set_command()
        # type 12 means return over.
        send_command([MadronaEvent(12,0 ,0, 0, 0, 0)])
        # receive_set_command()
    else:
        send_command([MadronaEvent(-1,0 ,0, 0, 0, 0)])
        # receive_set_command()
        
def check_result_noReponse_if_noResult():
    int_tensor=tensor_to_events(grid_world.madronaEventsResult)
    m_events=int_array_to_madrona_events(int_tensor)
    if len(m_events.events)>0 or len(flow_events_result)>0:
        for event in m_events.events:
            flow_events_result.append(event)
        grid_world.madronaEventsResult.copy_(empty_tensor())
        
        if len(flow_events_result)>0:
            first_element = flow_events_result.pop(0)
            send_command([first_element])
            # receive_set_command()
            return True
        # type 12 means return over.
        send_command([MadronaEvent(12,0 ,0, 0, 0, 0)])
        # receive_set_command()
        return True
    else:
        # send_command([MadronaEvent(-1,0 ,0, 0, 0, 0)])
        # receive_set_command()
        return False
def check_result_noReponse_if_returnContinue():
    int_tensor=tensor_to_events(grid_world.madronaEventsResult)
    m_events=int_array_to_madrona_events(int_tensor)
    if len(m_events.events)>0 or len(flow_events_result)>0:
        for event in m_events.events:
            flow_events_result.append(event)
        grid_world.madronaEventsResult.copy_(empty_tensor())
        
        if len(flow_events_result)>0:
            first_element = flow_events_result.pop(0)
            send_command([first_element])
            # receive_set_command()
        # type 12 means return over.
        else:
            send_command([MadronaEvent(12,0 ,0, 0, 0, 0)])
        # receive_set_command()
    else:
        send_command([MadronaEvent(12,0 ,0, 0, 0, 0)])
        # receive_set_command()


# to do
# memoryRW: read command and set field
def receive_set_command():
    global flow_events
    while True:
        events=comm.receive_data()
        m_events=events

        for event in m_events:
                # sim_send event
                if event.type==0:
                    flow_events.append(event) 
                    send_command([MadronaEvent(-1,0 ,0, 0, 0, 0)])
                    # receive_set_command()
                    break
                # step 1 event
                elif event.type==10:
                    if len(flow_events)>0:
                        int_tensor=tensor_to_events(grid_world.madronaEvents)
                        m_events=int_array_to_madrona_events(int_tensor)
                        # add 
                        for item in flow_events:
                            m_events.events.append(item)
                        flow_events=[]
                        # set back
                        madrona_events = MadronaEvents(m_events)
                        int_array = madrona_events_to_int_array(madrona_events)
                        int_tensor = events_to_tensor(int_array)
                        grid_world.madronaEvents.copy_(int_tensor)
                    
                    grid_world.step()
                    check_result_reponse()
                    break
                # step all event
                elif event.type==11:
                    while True:
                        if len(flow_events)>0:
                            int_tensor=tensor_to_events(grid_world.madronaEvents)
                            m_events=int_array_to_madrona_events(int_tensor)
                            # add 
                            for item in flow_events:
                                m_events.events.append(item)
                            flow_events=[]
                            # set back
                            madrona_events = MadronaEvents(m_events)
                            int_array = madrona_events_to_int_array(madrona_events)
                            int_tensor = events_to_tensor(int_array)
                            grid_world.madronaEvents.copy_(int_tensor)
                        grid_world.step()
                        if check_result_noReponse_if_noResult():
                            break
                    break
                # return continue
                elif event.type==13:
                        check_result_noReponse_if_returnContinue()
                        break
                # exit
                elif event.type==14:
                        send_command([])
                        sys.exit(0)
                        # break
                        
                
           
                

receive_set_command()
