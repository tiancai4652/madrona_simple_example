# set(SIMULATOR_SRCS
#     types.hpp sim.hpp build_topo.hpp sim_pfc.cpp fib_nexthop_init.hpp build_topo.cpp
# )

set(SIMULATOR_SRCS
    2_types_mflow.hpp  2_sim_mflow.hpp  2_build_topo_mflow.hpp  2_sim_mflow.cpp  2_fib_nexthop_init_mflow.hpp  2_build_topo_mflow.cpp
)

# file(GLOB SIMULATOR_SRCS 
#     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
#     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
# )

# set(SIMULATOR_SRCS
#     types_mflow.hpp sim_mflow.hpp build_topo_mflow.hpp sim_mflow.cpp fib_nexthop_init.hpp build_topo_mflow.cpp
# )

# set(BUILD_TOPO_SRCS   
#     build_topo.hpp build_topo.cpp 
# )



add_library(madrona_simple_ex_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

# add_library(madrona_simple_ex_cpu_impl STATIC
#     ${BUILD_TOPO_SRCS}
# )

target_link_libraries(madrona_simple_ex_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
)

add_library(madrona_simple_ex_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(madrona_simple_ex_mgr PRIVATE
    madrona_python_utils
    madrona_common
    madrona_importer
    madrona_mw_cpu
    madrona_simple_ex_cpu_impl
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            madrona_simple_ex_gpu_srcs
        SOURCES_DEFN
            SIMPLE_SRC_LIST
        FLAGS_DEFN
            SIMPLE_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madrona_simple_ex_mgr PRIVATE
        madrona_simple_ex_gpu_srcs
        madrona_mw_gpu
    )
endif()

target_compile_definitions(madrona_simple_ex_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(_madrona_simple_example_cpp
    bindings.cpp
)

target_link_libraries(_madrona_simple_example_cpp PRIVATE
    madrona_hdrs
    madrona_simple_ex_mgr
)
