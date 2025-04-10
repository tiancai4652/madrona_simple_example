#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>

#include "types.hpp"
#include "init.hpp"

using madrona::Entity;

namespace madsimple {

class Engine;

struct Sim : public madrona::WorldBase {
    struct Config {
        uint32_t maxEpisodeLength;
        bool enableViewer;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;
    const GridState *grid;
    uint32_t maxEpisodeLength;

    Entity init_entity;
    Entity chakra_nodes_entities[MAX_CHAKRA_NODES];
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
