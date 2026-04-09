#pragma once

#include "sim.hpp"

namespace madsimple {

extern const bool init_log_print_enabled;

void printInitTopoLog(const Sim &sim, Engine &ctx);
void printInitFlowLog(const Sim &sim);

}
