#include "sim.hpp"
#include "build_topo.hpp"

#include <madrona/mw_gpu_entry.hpp> 

namespace madsimple {

inline void _debug_e_port(Engine &ctx,
                        PortType port_type,
                        GlobalPortID &_global_port_id,
                        PktBuf &_queue);
}