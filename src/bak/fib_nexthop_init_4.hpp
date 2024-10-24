#include "sim.hpp"

#include <madrona/mw_gpu_entry.hpp> 

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>

#include "types.hpp"
#include "init.hpp"

#include <iostream>
#include <vector>
#include <algorithm>

// #include <cstdio>
// #include <cstring>
// #include <cstdlib>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

// Archtype: Switch
// SwitchType::edge
// SwitchType
// FIBTable
// calc_next_hop()
// const int32_t k_ary = 8; 
const int32_t num_switch = K_ARY*K_ARY*5/4;
const int32_t num_host = K_ARY*K_ARY*K_ARY/4;

const int32_t central_fib_table[num_switch][num_host][K_ARY] = {
{
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
},
{
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
},
{
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{2, 3, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
},
{
{0, -1, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
},
{
{0, -1, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
},
{
{0, -1, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
},
{
{0, -1, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{0, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{1, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{2, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
{3, -1, -1, -1},
},
};

const int32_t next_hop_table_host_to_sw[num_host] = {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29};



const int32_t next_hop_table[num_switch*K_ARY] = {0, 1, 32, 36, 2, 3, 33, 37, 4, 5, 40, 44, 6, 7, 41, 45, 8, 9, 48, 52, 10, 11, 49, 53, 12, 13, 56, 60, 14, 15, 57, 61, 2, 6, 64, 68, 3, 7, 72, 76, 10, 14, 65, 69, 11, 15, 73, 77, 18, 22, 66, 70, 19, 23, 74, 78, 26, 30, 67, 71, 27, 31, 75, 79, 34, 42, 50, 58, 35, 43, 51, 59, 38, 46, 54, 62, 39, 47, 55, 63};


}
