#pragma once

#include <madrona/components.hpp>

namespace madsimple {

enum class ExportID : uint32_t {
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    Results,
    Results2,
    SimulationTime,
    MadronaEvent,
    MadronaEvents,
    MadronaEventsResult,
    ProcessParams,
    NumExports
    
};

struct Reset {
    int32_t resetNow;
};

enum class Action : int32_t {
    Up    = 0,
    Down  = 1,
    Left  = 2,
    Right = 3,
    None,
};

struct GridPos {
    int32_t y;
    int32_t x;
};

struct Reward {
    float r;
};

struct Done {
    float episodeDone;
};
struct Results {
    uint32_t results;
};
struct Results2 {
   int32_t encoded_string[1000];
};

struct  MadronaEvent
{
    int32_t type;
    int32_t eventId;
    int32_t time;
    int32_t src;
    int32_t dst;
    int32_t size;
    int32_t port;
};

struct MadronaEventsQueue
{
    int32_t events[1000];
};

struct MadronaEvents
{
    int32_t events[1000];
};

struct  MadronaEventsResult
{
    int32_t events[1000];
};

struct  Configuration
{
    int32_t events[1000];
};

struct  ProcessParams
{
    int32_t params[1000];
};



struct CurStep {
    uint32_t step;
};

struct  SimulationTime
{
    int64_t time;
};


struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep,
    Results,
    Results2,
    SimulationTime,
    MadronaEventsQueue,
    MadronaEventsResult,
    MadronaEvents,
    ProcessParams
> {};

}
