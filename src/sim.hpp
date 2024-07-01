#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>

#include "types.hpp"
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
    CountT numInPort;
    CountT numEPort;

    Entity sender[K_ARY*K_ARY*K_ARY/4];
    Entity receiver[K_ARY*K_ARY*K_ARY/4];
    CountT numSender;
    CountT numReceiver;

    Entity _switch[K_ARY*K_ARY*5/4];
    CountT numSwitch;
    // CountT fib_table[2];
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
