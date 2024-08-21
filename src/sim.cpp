#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp> 


using namespace madrona;
using namespace madrona::math;

size_t custom_strlen(const char* str) {
    size_t len = 0;
    while (str[len] != '\0') {
        ++len;
    }
    return len;
}

namespace madsimple {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    registry.registerComponent<Results>();
    registry.registerComponent<Results2>();
    registry.registerComponent<SimulationTime>();
    registry.registerComponent<MadronaEventsQueue>();
    registry.registerComponent<MadronaEvents>();
    registry.registerComponent<MadronaEventsResult>();
    registry.registerComponent<ProcessParams>();
    registry.registerArchetype<Agent>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
    registry.exportColumn<Agent, Results>((uint32_t)ExportID::Results);
    registry.exportColumn<Agent, Results2>((uint32_t)ExportID::Results2);
    registry.exportColumn<Agent, SimulationTime>((uint32_t)ExportID::SimulationTime);
    registry.exportColumn<Agent, MadronaEvents>((uint32_t)ExportID::MadronaEvents);
    registry.exportColumn<Agent, MadronaEventsResult>((uint32_t)ExportID::MadronaEventsResult);
     registry.exportColumn<Agent, ProcessParams>((uint32_t)ExportID::ProcessParams);
    
}

const int INTS_PER_EVENT = 7;  // 每个事件7个整数
int parseMadronaEvents(const MadronaEvents &madronaEvents, MadronaEvent parsedEvents[], int maxEvents) {
    int validEvents = 0;
    for (int i = 0; i < maxEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex + INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        MadronaEvent event = {
            madronaEvents.events[baseIndex],
            madronaEvents.events[baseIndex + 1],
            madronaEvents.events[baseIndex + 2],
            madronaEvents.events[baseIndex + 3],
            madronaEvents.events[baseIndex + 4],
            madronaEvents.events[baseIndex + 5],
            madronaEvents.events[baseIndex + 6]
        };
        // 如果解析出的事件全为0,则忽略
        if (event.type == 0 && event.eventId == 0 &&
            event.time == 0 && event.src == 0 &&
            event.dst == 0 && event.size == 0 && event.port == 0) {
            continue;
        }
        parsedEvents[validEvents++] = event;
    }
    return validEvents;  // 返回有效事件的数量
}
void updateMadronaEvents(MadronaEvents &madronaEvents, const MadronaEvent parsedEvents[], int numEvents) {
    // 清空 madronaEvents
    for (int i = 0; i < 1000; ++i) {
        madronaEvents.events[i] = 0;
    }

    for (int i = 0; i < numEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex +  INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        madronaEvents.events[baseIndex] = parsedEvents[i].type;
        madronaEvents.events[baseIndex + 1] = parsedEvents[i].eventId;
        madronaEvents.events[baseIndex + 2] = parsedEvents[i].time;
        madronaEvents.events[baseIndex + 3] = parsedEvents[i].src;
        madronaEvents.events[baseIndex + 4] = parsedEvents[i].dst;
        madronaEvents.events[baseIndex + 5] = parsedEvents[i].size;
        madronaEvents.events[baseIndex + 6] = parsedEvents[i].port;
    }
}  

int parseMadronaEvents(const MadronaEventsResult &madronaEvents, MadronaEvent parsedEvents[], int maxEvents) {
    int validEvents = 0;
    for (int i = 0; i < maxEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex + INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        MadronaEvent event = {
            madronaEvents.events[baseIndex],
            madronaEvents.events[baseIndex + 1],
            madronaEvents.events[baseIndex + 2],
            madronaEvents.events[baseIndex + 3],
            madronaEvents.events[baseIndex + 4],
            madronaEvents.events[baseIndex + 5],
            madronaEvents.events[baseIndex + 6]
        };
        // 如果解析出的事件全为0,则忽略
        if (event.type == 0 && event.eventId == 0 &&
            event.time == 0 && event.src == 0 &&
            event.dst == 0 && event.size == 0 && event.port == 0) {
            continue;
        }
        parsedEvents[validEvents++] = event;
    }
    return validEvents;  // 返回有效事件的数量
}
void updateMadronaEvents(MadronaEventsResult &madronaEvents, const MadronaEvent parsedEvents[], int numEvents) {
    // 清空 madronaEvents
    for (int i = 0; i < 1000; ++i) {
        madronaEvents.events[i] = 0;
    }

    for (int i = 0; i < numEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex +  INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        madronaEvents.events[baseIndex] = parsedEvents[i].type;
        madronaEvents.events[baseIndex + 1] = parsedEvents[i].eventId;
        madronaEvents.events[baseIndex + 2] = parsedEvents[i].time;
        madronaEvents.events[baseIndex + 3] = parsedEvents[i].src;
        madronaEvents.events[baseIndex + 4] = parsedEvents[i].dst;
        madronaEvents.events[baseIndex + 5] = parsedEvents[i].size;
        madronaEvents.events[baseIndex + 6] = parsedEvents[i].port;
    }
}

int parseMadronaEvents(const MadronaEventsQueue &madronaEvents, MadronaEvent parsedEvents[], int maxEvents) {
    int validEvents = 0;
    for (int i = 0; i < maxEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex + INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        MadronaEvent event = {
            madronaEvents.events[baseIndex],
            madronaEvents.events[baseIndex + 1],
            madronaEvents.events[baseIndex + 2],
            madronaEvents.events[baseIndex + 3],
            madronaEvents.events[baseIndex + 4],
            madronaEvents.events[baseIndex + 5],
            madronaEvents.events[baseIndex + 6]
        };
        // 如果解析出的事件全为0,则忽略
        if (event.type == 0 && event.eventId == 0 &&
            event.time == 0 && event.src == 0 &&
            event.dst == 0 && event.size == 0 && event.port == 0) {
            continue;
        }
        parsedEvents[validEvents++] = event;
    }
    return validEvents;  // 返回有效事件的数量
}
void updateMadronaEvents(MadronaEventsQueue &madronaEvents, const MadronaEvent parsedEvents[], int numEvents) {
    // 清空 madronaEvents
    for (int i = 0; i < 1000; ++i) {
        madronaEvents.events[i] = 0;
    }

    for (int i = 0; i < numEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex +  INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        madronaEvents.events[baseIndex] = parsedEvents[i].type;
        madronaEvents.events[baseIndex + 1] = parsedEvents[i].eventId;
        madronaEvents.events[baseIndex + 2] = parsedEvents[i].time;
        madronaEvents.events[baseIndex + 3] = parsedEvents[i].src;
        madronaEvents.events[baseIndex + 4] = parsedEvents[i].dst;
        madronaEvents.events[baseIndex + 5] = parsedEvents[i].size;
        madronaEvents.events[baseIndex + 6] = parsedEvents[i].port;
    }
}



inline void tick(Engine &ctx,
                 Action &action,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step,
                 Results &results,
                 Results2 &results2,
                 // SimulationTime
                 SimulationTime &time,
                 // internal event queue to save event excuted in future 
                 MadronaEventsQueue &madronaEventsQueue,
                 // external event to put in madronaEventsQueue
                 MadronaEvents &madronaEvents,
                 // event result at current time
                 MadronaEventsResult &madronaEventsResult,
                 ProcessParams &processParams
                 )
{


    printf("gpu:\n");
    printf("schedule run: %d\n", processParams.params[0]);
    printf("flow run: %d\n", processParams.params[1]);
    printf("simulation_time: %ld\n",time.time);

    

// test
// for (int i = 0; i < 6; i++)
// {
//     printf("%d",madronaEvents.events[i]);
// }

    printf("parse madronaEvents\n");
    const int maxEvents = 1000 / 7;
    MadronaEvent parsedEvents[maxEvents];
    // parse MadronaEvents
    int validEvents = parseMadronaEvents(madronaEvents, parsedEvents, maxEvents);
    if (validEvents == 0)
    {
        printf("MadronaEvents is empty!\n ");
    }
    // add madronaEvents to madronaEventsQueue
    else
    {
        printf("receiveEvents : %d\n",validEvents);
        // print events
        for (int i = 0; i < validEvents; ++i)
        {
            printf("Event %d: type=%d, eventId=%d, time=%d, src=%d, dst=%d, size=%d, port=%d\n",
                   i, parsedEvents[i].type, parsedEvents[i].eventId,
                   parsedEvents[i].time, parsedEvents[i].src,
                   parsedEvents[i].dst, parsedEvents[i].size,parsedEvents[i].port);
        }
        MadronaEvent exitedEvents[maxEvents];
        // add receive events to event queue.
        int existedEventsNum = parseMadronaEvents(madronaEventsQueue, exitedEvents, maxEvents);
        printf("eventsQueueNum : %d\n",existedEventsNum);
        for (int i = 0; i < validEvents; ++i)
        {
            // relative time changes to absolute time
            parsedEvents[i].time+=time.time;
            exitedEvents[existedEventsNum+i]=parsedEvents[i];
        }
        updateMadronaEvents(madronaEventsQueue,exitedEvents,existedEventsNum+validEvents);
        // clear recieve events.
        updateMadronaEvents(madronaEvents, parsedEvents, 0);
        // // for instance: change one event property
        // if (validEvents > 0)
        // {
        //     parsedEvents[0].time += 1;
        // }
        // updateMadronaEvents(madronaEvents, parsedEvents, validEvents);
    }

        // 1 frame 1 ns
    time.time = time.time + 1;

    MadronaEvent eventsQueue[maxEvents];
    // process event queue
    int eventsQueueNum = parseMadronaEvents(madronaEventsQueue, eventsQueue, maxEvents);
    printf("eventsQueueNum :%d\n",eventsQueueNum);
    MadronaEvent eventsResult[maxEvents];
    int resultIndex = 0;
    MadronaEvent eventsFuture[maxEvents];
    int futureIndex = 0;
    for (int i = 0; i < eventsQueueNum; i++)
    {
        // printf("time.time:%ld\n",time.time);
        printf("eventsQueue[%d].time:%d,type:%d,id:%d\n",i,eventsQueue[i].time,eventsQueue[i].type,eventsQueue[i].eventId);
        // to do :maybe relative time,done
        if (time.time >= eventsQueue[i].time)
        {
            // sim_schedule event.
            if (eventsQueue[i].type == 2)
            {
                eventsResult[resultIndex] = eventsQueue[i];
                // scheduel not to set time
                // eventsResult[resultIndex].time=time.time;
                resultIndex++;
                printf("process sim_schedule event. time: %d\n", eventsResult[resultIndex].time);
            }
            // sim_send event.
            else if (eventsQueue[i].type == 0)
            {
                // assume flow finish at 20 frame after.
                if (time.time >= eventsQueue[i].time + 20)
                {
                    // to do : 1 set flow entities
                    // to do : 2 set result to eventsResult when flow done
                    eventsResult[resultIndex] = eventsQueue[i];
                    eventsResult[resultIndex].time = eventsQueue[i].time + 20;
                    resultIndex++;
                    printf("process sim_send event,time:%ld.\n", time.time);
                }
                else
                {
                    eventsFuture[futureIndex] = eventsQueue[i];
                    futureIndex++;
                }
            }
        }
        else
        {
            eventsFuture[futureIndex] = eventsQueue[i];
            futureIndex++;
        }
    }

    // put to madronaEventsResult
    if (resultIndex > 0)
    {
        printf("resultNum:%d\n", resultIndex + 1);
        updateMadronaEvents(madronaEventsResult, eventsResult, resultIndex + 1);
    }

    printf("futureNum:%d\n", futureIndex + 1);
    updateMadronaEvents(madronaEventsQueue, eventsFuture, futureIndex + 1);

    // printf("tick2\n");
    // for (int i = 0; i < 6; i++)
    // {
    //     printf("%d\n", results2.encoded_string[i]);
    // }

    // const char* new_string = "updated";
    // size_t new_string_length = custom_strlen(new_string);
    // // 找到已有字符串的长度
    // size_t current_length = 0;
    // while (current_length < 1000 && results2.encoded_string[current_length] != 0) {
    //     current_length++;
    // }

    // // 确保新字符串不会超过最大长度
    // size_t available_space = 1000 - current_length;

    // for (size_t i = 0; i < new_string_length && i < available_space; ++i) {
    //     results2.encoded_string[current_length + i] = static_cast<int32_t>(new_string[i]);
    // }

    // // 如果有空间，添加终止符
    // if (current_length + new_string_length < 1000) {
    //     results2.encoded_string[current_length + new_string_length] = 0; // 终止符
    // }

    // const GridState *grid = ctx.data().grid;

    // GridPos new_pos = grid_pos;

    // switch (action) {
    //     case Action::Up: {
    //         new_pos.y += 1;
    //     } break;
    //     case Action::Down: {
    //         new_pos.y -= 1;
    //     } break;
    //     case Action::Left: {
    //         new_pos.x -= 1;
    //     } break;
    //     case Action::Right: {
    //         new_pos.x += 1;
    //     } break;
    //     default: break;
    // }

    // action = Action::None;

    // if (new_pos.x < 0) {
    //     new_pos.x = 0;
    // }

    // if (new_pos.x >= grid->width) {
    //     new_pos.x = grid->width - 1;
    // }

    // if (new_pos.y < 0) {
    //     new_pos.y = 0;
    // }

    // if (new_pos.y >= grid->height) {
    //     new_pos.y = grid->height -1;
    // }


    // {
    //     const Cell &new_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

    //     if ((new_cell.flags & CellFlag::Wall)) {
    //         new_pos = grid_pos;
    //     }
    // }

    // const Cell &cur_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

    // bool episode_done = false;
    // if (reset.resetNow != 0) {
    //     reset.resetNow = 0;
    //     episode_done = true;
    // }

    // if ((cur_cell.flags & CellFlag::End)) {
    //     episode_done = true;
    // }

    // uint32_t cur_step = episode_step.step;

    // if (cur_step == ctx.data().maxEpisodeLength - 1) {
    //     episode_done = true;
    // }
    // results.results=results.results+1;
    // // printf("sim****\n");
    // // printf("%d", results.results);
    // // printf("sim****\n");
    // if (episode_done) {
    //     done.episodeDone = 1.f;

    //     new_pos = GridPos {
    //         grid->startY,
    //         grid->startX,
    //     };

    //     episode_step.step = 0;
    // } else {
    //     done.episodeDone = 0.f;
    //     episode_step.step = cur_step + 1;
    // }

    // // Commit new position
    // grid_pos = new_pos;
    // reward.r = cur_cell.reward;
}



void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
{
    printf("*******Enter into setupTasks*********\n");
    builder.addToGraph<ParallelForNode<Engine, tick, Action, Reset, GridPos, Reward, Done, CurStep,Results,Results2,SimulationTime,MadronaEventsQueue,MadronaEvents,MadronaEventsResult,ProcessParams>>({});
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    printf("Sim Constructorsssss\n");
    Entity agent = ctx.makeEntity<Agent>();
    ctx.get<Action>(agent) = Action::None;
    ctx.get<GridPos>(agent) = GridPos {
        grid->startY,
        grid->startX,
    };
    ctx.get<Reward>(agent).r = 0.f;
    ctx.get<Done>(agent).episodeDone = 0.f;
    ctx.get<CurStep>(agent).step = 0;
    ctx.get<Results>(agent).results = 0.f;
    ctx.get<SimulationTime>(agent).time = 0;
    // ctx.get<MadronaEvents>(agent).events[0].type = 1;
    printf("%d\n",ctx.get<MadronaEvents>(agent).events[0]);
    //  printf("Sim Constructorsssss end\n");
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
