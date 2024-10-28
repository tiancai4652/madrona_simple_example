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


schedule_events=[]
flow_events=[]
schedule_num=-1
flow_num=-1

schedule_events_result=[]
flow_events_result=[]

schedule_send_back=False
flow_send_back=False

time_current=None

# memoryRW: set params string, for instance
example_string = ""
encoded_string_tensor = string_to_tensor(example_string)
grid_world.results2.copy_(encoded_string_tensor)

# to do
# memoryRW: read command and set field
def receive_set_command():
    global schedule_events,flow_events,schedule_num,flow_num
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
    # for event in m_events:
    #     event.print_event()
    # process one event per time
    for event in m_events:
        # set scheduel event num
        # event.src in torch[0] is schedule num.
        if event.type==10:
            schedule_num=event.src
            flow_num=event.src
            events = [MadronaEvent(0,0 ,0, 0, 0, 0)]
            send_command(events)
            receive_set_command()
        # event.src in torch[0] is schedule num.
        # elif event.type==20:
        #     flow_num=event.src
        #     events = [MadronaEvent(0,0 ,0, 0, 0, 0)]
        #     send_command(events)
        #     receive_set_command()
        # firsr to process sim_get_time event, type is 3
        elif event.type==3:
            time=grid_world.simulation_time.cpu().item()
            if time_current is not None:
                time=time_current
            events = [MadronaEvent(event.type, event.eventId, time, 0, 0, 0)]
            send_command(events)
            receive_set_command()
        # second, we process sim_schedule and sim_send, type is 2,0
        elif event.type==2 or event.type==0:
            if event.type==2:
                schedule_events.append(event)
            if event.type==0:
                flow_events.append(event) 
            
            # tell madrona to run next frames until get result.
            if event.type==2 and len(schedule_events)>=schedule_num and schedule_num!=-1:
            # if len(schedule_events)>=schedule_num :
                # read event buffer
                int_tensor=tensor_to_events(grid_world.madronaEvents)
                m_events=int_array_to_madrona_events(int_tensor)
                # add 
                for item in schedule_events:
                    m_events.events.append(item)
                schedule_events=[]
                # set back
                madrona_events = MadronaEvents(m_events)
                int_array = madrona_events_to_int_array(madrona_events)
                int_tensor = events_to_tensor(int_array)
                grid_world.madronaEvents.copy_(int_tensor)
                # schedule_num=-1
                
                # index=0 means run scheduel 
                int_tensor = events_to_tensor([1,0])
                grid_world.processParams.copy_(int_tensor)
                # int_tensor=tensor_to_events(grid_world.madronaEvents)
                # m_events=int_array_to_madrona_events(int_tensor)
                # for event in m_events.events:
                #     event.print_event()
                # assume one event per time.
                if not schedule_send_back:
                    grid_world.step()

            # tell madrona to run next frames until get result.
            elif event.type==0 and len(flow_events)>=flow_num and flow_num!=-1:
            # elif len(flow_events)>=flow_num :
                # read event buffer
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
                # flow_num=-1
                
                # index=0 means run scheduel 
                int_tensor = events_to_tensor([0,1])
                grid_world.processParams.copy_(int_tensor)
                # int_tensor=tensor_to_events(grid_world.madronaEvents)
                # m_events=int_array_to_madrona_events(int_tensor)
                # for event in m_events.events:
                #     event.print_event()
                # assume one event per time.
                if not flow_send_back:
                    grid_world.step()
            else:
                events = [MadronaEvent(0,0 ,0, 0, 0, 0)]
                send_command(events)
                receive_set_command()
        #  third, "event type == -100" means nothing. for go next frame.
        elif event.type==-100:
            # assume one event per time.
            # grid_world.step()
            break
            
            

def send_command(events):
    comm.send_data(events)


# for i in range(5):
while True:    
    print("cpu:")
    
    print("time:")
    print(grid_world.simulation_time.cpu().item())
    
    # not finish
    int_tensor=tensor_to_events(grid_world.processParams)
    in_scheduel=int_tensor[0][0]==1
    in_flow=int_tensor[0][1]==1
    if(in_scheduel or in_flow):
        # send_command()
        int_tensor=tensor_to_events(grid_world.madronaEventsResult)
        m_events=int_array_to_madrona_events(int_tensor)
        if (in_scheduel and len(m_events.events)==schedule_num) or schedule_send_back:
            schedule_send_back=True
            for event in m_events.events:
                if event.type==2:
                    schedule_events_result.append(event)
            grid_world.madronaEventsResult.copy_(empty_tensor())
            # grid_world.processParams.copy_(empty_tensor())
            # schedule_num=-1
            index=1
             
            for event in schedule_events_result:
                time_current=event.time
                print(f"process {index} result")
                index=index+1
                send_command([event])
                receive_set_command()
            schedule_send_back=False
            schedule_events_result=[]
            time_current=None
        elif (in_flow and len(m_events.events)==flow_num) or flow_send_back:
            flow_send_back=True
            for event in m_events.events:
                if event.type==0:
                    flow_events_result.append(event)
            grid_world.madronaEventsResult.copy_(empty_tensor())
            # grid_world.processParams.copy_(empty_tensor())
            # flow_num=-1
 
            for event in flow_events_result:
                time_current=event.time
                send_command([event])
                receive_set_command()
            flow_send_back=False
            flow_events_result=[]
            time_current=None
        else:
            grid_world.step()     
    else:
        receive_set_command()
        
        
        # if len(m_events.events)>0:
        #     # for event in m_events.events:
        #     #     event.print_event()
        #     for event in m_events.events:
        #         send_command([event])
        #         receive_set_command()
        #         # remove 0
        #         m_events.events[0].set_empty()
        #         # set back
        #         madrona_events = MadronaEvents(m_events)
        #         int_array = madrona_events_to_int_array(madrona_events)
        #         int_tensor = events_to_tensor(int_array)
        #         grid_world.madronaEventsResult.copy_(int_tensor)
        
    # else:
    #     send_command([])
    # to do
    # check if end
    # quit()
    
    # print()
