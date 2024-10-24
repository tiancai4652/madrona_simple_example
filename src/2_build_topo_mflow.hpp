#include "2_sim_mflow.hpp"

#include <madrona/mw_gpu_entry.hpp> 

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>

#include "2_types_mflow.hpp"
#include "init.hpp"

namespace madsimple {

using madrona::Entity;
using madrona::CountT; 

void generate_switch(Engine &ctx, CountT k_ary);

void generate_in_port(Engine &ctx, CountT k_ary);

void generate_e_port(Engine &ctx, CountT k_ary);

void generate_host(Engine &ctx, CountT k_ary);

}