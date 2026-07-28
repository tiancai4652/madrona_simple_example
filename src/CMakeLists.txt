set(SIMULATOR_SRCS
    types.hpp sim.hpp
    sim.cpp
    sim_tasks.cpp
    sim_init.cpp
    sim_init_state.cpp
    sim_init_port.cpp
    sim_init_topo.cpp
    sim_init_npu.cpp
    sim_world.cpp
    sim_debug.cpp
    sim_systems.cpp
    sim_systems_lifecycle.cpp
    sim_systems_schedule.cpp
    sim_systems_dispatch.cpp
    sim_systems_progress.cpp
    sim_systems_ingress.cpp
    sim_systems_bandwidth.cpp
    sim_systems_buffer.cpp
    sim_systems_pfc.cpp
    sys/net_sys_interface.cpp
    sys/fake_system.cpp
)

add_library(madrona_simple_ex_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

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

target_include_directories(_madrona_simple_example_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(_madrona_simple_example_cpp PRIVATE
    madrona_hdrs
    madrona_simple_ex_mgr
)
