#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>

#include "types_mflow.hpp"
#include "init.hpp"

namespace madsimple {

using madrona::Entity;
using madrona::CountT;

class Engine;

struct Sim : public madrona::WorldBase {
    struct Config {
        uint32_t maxEpisodeLength;
        bool enableViewer;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphBuilder &builder,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;
    const GridState *grid;
    uint32_t maxEpisodeLength;

    // const CountT k_ary;
    //Entity inPorts[(4*4*5/4)*4];  //4*4*5/4)*4
    //Entity ePorts[(4*4*5/4)*4];  //4*4*5/4)*4
    Entity inPorts[(K_ARY*K_ARY*5/4)*K_ARY]; 
    Entity ePorts[(K_ARY*K_ARY*5/4)*K_ARY];
    uint32_t numInPort;
    uint32_t numEPort;

    // Entity senders[K_ARY*K_ARY*K_ARY/4];
    // Entity receivers[K_ARY*K_ARY*K_ARY/4];
    // uint32_t numSender;
    // uint32_t numReceiver;

    Entity _switches[K_ARY*K_ARY*5/4];
    uint32_t numSwitch;

    Entity _snd_flows[2*K_ARY*K_ARY*K_ARY/4];
    Entity _recv_flows[2*K_ARY*K_ARY*K_ARY/4];
    uint32_t num_snd_flow;
    uint32_t num_recv_flow;   

    Entity _nics[K_ARY*K_ARY*K_ARY*4];
    uint32_t num_nic;

    Entity _npus[K_ARY*K_ARY*K_ARY*4];
    uint32_t num_npu;  

    // CountT fib_table[2];
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
