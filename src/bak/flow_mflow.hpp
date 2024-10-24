// #include "sim.hpp"

// #include <madrona/mw_gpu_entry.hpp> 

// #include <madrona/taskgraph_builder.hpp>
// #include <madrona/custom_context.hpp>

// #include "types.hpp"
// #include "init.hpp"

using namespace madrona;

namespace madsimple {

// const int32_t num_switch = K_ARY*K_ARY*5/4;
// const int32_t num_host = K_ARY*K_ARY*K_ARY/4;

// const int32_t fows[2] = {
//     {0, 2, 14600},
//     {1, 3, 14600}
// }
const int32_t flow_events[2][8] = {
//{flow_id, src, dst, flow_size, start_time, nic_id, snd_server_id, recv_server_id},
    {0, 0, 8, 2000*1000, 0, 0, 0, 8}, 
    {1, 0, 9, 2000*1000, 0, 0, 0, 9},
};

}
