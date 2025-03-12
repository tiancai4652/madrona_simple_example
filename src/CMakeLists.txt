# 在文件开始处添加include路径定义
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sys_layer
    ${CMAKE_CURRENT_SOURCE_DIR}/sys_layer/astra-sim
    ${CMAKE_CURRENT_SOURCE_DIR}/sys_layer/astra-sim/network_frontend/analytical
    ${CMAKE_CURRENT_SOURCE_DIR}/sys_layer/astra-sim/containers
)


set(SIMULATOR_SRCS
    types.hpp sim.hpp sim.cpp
)

# 添加系统层源文件
set(SYS_LAYER_SRCS
    sys_layer/Astrasim.hpp
    sys_layer/Astrasim.cpp
    sys_layer/sysComponent.hpp
    sys_layer/HelperTool.hpp
    sys_layer/HelperTool.cpp
    sys_layer/containers/FixedTuple.hpp
    sys_layer/containers/FixedMap.hpp
    sys_layer/containers/Optional.hpp
    sys_layer/containers/Pair.hpp
    sys_layer/containers/CustomMap.hpp

    sys_layer/astra-sim/network_frontend/analytical/common/Event.hpp
    sys_layer/astra-sim/network_frontend/analytical/common/Event.cpp
    sys_layer/astra-sim/network_frontend/analytical/common/Type.hpp

    
    sys_layer/astra-sim/network_frontend/analytical/common/ChunkIdGeneratorEntry.cpp
    sys_layer/astra-sim/network_frontend/analytical/common/ChunkIdGeneratorEntry.hpp
    sys_layer/astra-sim/network_frontend/analytical/common/ChunkIdGenerator.cpp
    sys_layer/astra-sim/network_frontend/analytical/common/CallbackTrackerEntry.cpp
    sys_layer/astra-sim/network_frontend/analytical/common/CallbackTrackerEntry.hpp
)

add_library(madrona_simple_ex_cpu_impl STATIC
    ${SIMULATOR_SRCS}
    ${SYS_LAYER_SRCS}
)

target_include_directories(madrona_simple_ex_cpu_impl
    PUBLIC
        ${INCLUDE_DIRS}
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
        INCLUDE_DIRS
            ${INCLUDE_DIRS}
        SRCS
            ${SIMULATOR_SRCS}
            ${SYS_LAYER_SRCS}
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

