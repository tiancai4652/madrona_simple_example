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

    Entity inPorts[(K_ARY*K_ARY*5/4)*K_ARY]; 
    Entity ePorts[(K_ARY*K_ARY*5/4)*K_ARY];
    uint32_t numInPort;
    uint32_t numEPort;

    Entity senders[K_ARY*K_ARY*K_ARY/4];
    Entity receivers[K_ARY*K_ARY*K_ARY/4];
    uint32_t numSender;
    uint32_t numReceiver;

    Entity _switches[K_ARY*K_ARY*5/4];
    uint32_t numSwitch;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
