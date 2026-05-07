#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
using namespace madrona;
using namespace madrona::py;

namespace madsimple {

namespace {

inline uint64_t alignOffset(uint64_t offset, uint64_t alignment)
{
    uint64_t remainder = offset % alignment;
    if (remainder == 0) {
        return offset;
    }
    return offset + (alignment - remainder);
}

struct NetworkLayout {
    uint64_t totalBytes;
    uint64_t nodesOffset;
    uint64_t linksOffset;
    uint64_t flowsOffset;
};

inline NetworkLayout getNetworkLayout(const NetworkInit &src_network)
{
    uint64_t offset = sizeof(NetworkInit);
    offset = alignOffset(offset, alignof(NodeDef));
    uint64_t nodes_offset = offset;
    offset += sizeof(NodeDef) * (uint64_t)src_network.numNodes;

    offset = alignOffset(offset, alignof(LinkDef));
    uint64_t links_offset = offset;
    offset += sizeof(LinkDef) * (uint64_t)src_network.numLinks;

    offset = alignOffset(offset, alignof(FlowDef));
    uint64_t flows_offset = offset;
    offset += sizeof(FlowDef) * (uint64_t)src_network.numFlows;

    return NetworkLayout {
        .totalBytes = offset,
        .nodesOffset = nodes_offset,
        .linksOffset = links_offset,
        .flowsOffset = flows_offset,
    };
}

inline void sortFlowsByStartTime(FlowDef *flows, int32_t num_flows)
{
    if (num_flows <= 1) {
        return;
    }

    std::stable_sort(flows, flows + num_flows,
        [](const FlowDef &a, const FlowDef &b) {
            return a.start_time < b.start_time;
    });
}

inline bool envFlagValue(const char *env)
{
    return !(env[0] == '0' && env[1] == '\0');
}

inline bool envFlagEnabled(const char *name)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return false;
    }

    return envFlagValue(env);
}

inline int32_t resolvePerfFCTOnly(const Manager::Config &cfg)
{
    const char *override_env = std::getenv("MADSIMPLE_PERF_FCT_ONLY");
    if (override_env != nullptr && override_env[0] != '\0') {
        return envFlagValue(override_env) ? 1 : 0;
    }

    if (envFlagEnabled("init_log_print_enabled") ||
        envFlagEnabled("system_log_print_enabled")) {
        return 0;
    }

    return cfg.perf_fct_only != 0 ? 1 : 0;
}

#ifdef MADRONA_CUDA_SUPPORT
inline CompileConfig::OptMode getGPUOptMode()
{
    const char *env = std::getenv("MADRONA_MWGPU_OPT_MODE");
    if (env == nullptr || env[0] == '\0') {
        return CompileConfig::OptMode::LTO;
    }

    if (std::strcmp(env, "optimize") == 0 ||
        std::strcmp(env, "opt") == 0) {
        return CompileConfig::OptMode::Optimize;
    }

    if (std::strcmp(env, "debug") == 0) {
        return CompileConfig::OptMode::Debug;
    }

    if (std::strcmp(env, "lto") == 0) {
        return CompileConfig::OptMode::LTO;
    }

    return CompileConfig::OptMode::LTO;
}
#endif

} // namespace

struct Manager::Impl {
    Config cfg;
    EpisodeManager *episodeMgr;
    GridState *gridData;
    NetworkInit *networkData;

    inline Impl(const Config &c,
                EpisodeManager *ep_mgr,
                GridState *grid_data,
                NetworkInit *network_data)
        : cfg(c),
          episodeMgr(ep_mgr),
          gridData(grid_data),
          networkData(network_data)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;
    virtual Tensor exportTensor(ExportID slot, TensorElementType type,
                                Span<const int64_t> dims) = 0;
    virtual double simulationTime() = 0;
    virtual int32_t numFlowDefs() = 0;
    virtual int32_t numPendingFlows() = 0;
    virtual int32_t numDelayedEvents() = 0;
    virtual int32_t numActiveTags() = 0;
    virtual int32_t numSourceTags() = 0;
    virtual int32_t numFlowCompletions() = 0;
    virtual FlowCompletionRecord flowCompletion(int32_t idx) = 0;
    virtual StepPhaseTimes lastStepPhaseTimes() = 0;

    static inline Impl * init(const Config &cfg,
                              const GridState &src_grid,
                              const NetworkInit &src_network);
};

struct Manager::CPUImpl final : Manager::Impl {
    using ExecT = TaskGraphExecutor<Engine, Sim, Sim::Config, WorldInit>;
    ExecT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   GridState *grid_data,
                   NetworkInit *network_data,
                   WorldInit *world_inits)
        : Impl(mgr_cfg, episode_mgr, grid_data, network_data),
          cpuExec({
                  .numWorlds = mgr_cfg.numWorlds,
                  .numExportedBuffers = (uint32_t)ExportID::NumExports,
              }, sim_cfg, world_inits, 1)
    {}

    inline virtual ~CPUImpl() final {
        delete episodeMgr;
        free(gridData);
        free(networkData);
    }

    inline virtual void run() final { cpuExec.run(); }

    inline virtual Tensor exportTensor(ExportID slot,
                                       TensorElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }

    inline SimStats &fetchSimStats()
    {
        auto *stats = (SimStats *)cpuExec.getExported(
            (uint32_t)ExportID::SimStats);
        return stats[0];
    }

    inline virtual double simulationTime() final
    {
        return fetchSimStats().simulationTime;
    }

    inline virtual int32_t numFlowDefs() final
    {
        return fetchSimStats().numFlowDefs;
    }

    inline virtual int32_t numPendingFlows() final
    {
        return fetchSimStats().numPendingFlows;
    }

    inline virtual int32_t numDelayedEvents() final
    {
        return fetchSimStats().numDelayedEvents;
    }

    inline virtual int32_t numActiveTags() final
    {
        return fetchSimStats().numActiveTags;
    }

    inline virtual int32_t numSourceTags() final
    {
        return fetchSimStats().numSourceTags;
    }

    inline virtual int32_t numFlowCompletions() final
    {
        return fetchSimStats().numFlowCompletions;
    }

    inline virtual FlowCompletionRecord flowCompletion(int32_t idx) final
    {
        if (idx < 0 || idx >= fetchSimStats().numFlowCompletions) {
            return FlowCompletionRecord {};
        }
        auto *buf = (FlowCompletionBuf *)cpuExec.getExported(
            (uint32_t)ExportID::FlowCompletionBuf);
        return buf[0].records[idx];
    }

    inline virtual StepPhaseTimes lastStepPhaseTimes() final
    {
        auto *times = (StepPhaseTimes *)cpuExec.getExported(
            (uint32_t)ExportID::StepPhaseTimes);
        return times[0];
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::GPUImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;

    inline GPUImpl(CUcontext cu_ctx,
                   const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   GridState *grid_data,
                   NetworkInit *network_data,
                   WorldInit *world_inits)
        : Impl(mgr_cfg, episode_mgr, grid_data, network_data),
          gpuExec({
                  .worldInitPtr = world_inits,
                  .numWorldInitBytes = sizeof(WorldInit),
                  .userConfigPtr = (void *)&sim_cfg,
                  .numUserConfigBytes = sizeof(Sim::Config),
                  .numWorldDataBytes = sizeof(Sim),
                  .worldDataAlignment = alignof(Sim),
                  .numWorlds = mgr_cfg.numWorlds,
                  .numTaskGraphs = 1,
                  .numExportedBuffers = (uint32_t)ExportID::NumExports,
              }, {
                  { SIMPLE_SRC_LIST },
                  { SIMPLE_COMPILE_FLAGS },
                  getGPUOptMode(),
              }, cu_ctx),
          stepGraph(gpuExec.buildLaunchGraph(0))

    {}

    inline virtual ~GPUImpl() final {
        REQ_CUDA(cudaFree(episodeMgr));
        REQ_CUDA(cudaFree(gridData));
        REQ_CUDA(cudaFree(networkData));
    }

    inline virtual void run() final {
        gpuExec.run(stepGraph);
    }

    virtual inline Tensor exportTensor(ExportID slot, TensorElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }

    // Host-side snapshot of the per-step SimStats mirror that
    // updateSimStatsStepSystem writes into the SimDriverArch singleton
    // at the end of every task graph. Refreshed on every getter call via
    // a synchronous cudaMemcpy of sizeof(SimStats) (~32 bytes) from the
    // GPU exported column. This latency is only paid on Python-visible
    // inspection calls, never inside the hot step loop on GPU.
    inline SimStats fetchSimStats()
    {
        auto *dev_ptr = (SimStats *)gpuExec.getExported(
            (uint32_t)ExportID::SimStats);
        SimStats host_stats {};
        REQ_CUDA(cudaMemcpy(&host_stats, dev_ptr, sizeof(SimStats),
                            cudaMemcpyDeviceToHost));
        return host_stats;
    }

    inline virtual double simulationTime() final
    {
        return fetchSimStats().simulationTime;
    }

    inline virtual int32_t numFlowDefs() final
    {
        return fetchSimStats().numFlowDefs;
    }

    inline virtual int32_t numPendingFlows() final
    {
        return fetchSimStats().numPendingFlows;
    }

    inline virtual int32_t numDelayedEvents() final
    {
        return fetchSimStats().numDelayedEvents;
    }

    inline virtual int32_t numActiveTags() final
    {
        return fetchSimStats().numActiveTags;
    }

    inline virtual int32_t numSourceTags() final
    {
        return fetchSimStats().numSourceTags;
    }

    inline virtual int32_t numFlowCompletions() final
    {
        return fetchSimStats().numFlowCompletions;
    }

    inline virtual FlowCompletionRecord flowCompletion(int32_t idx) final
    {
        int32_t n = fetchSimStats().numFlowCompletions;
        if (idx < 0 || idx >= n) {
            return FlowCompletionRecord {};
        }
        auto *dev_buf = (FlowCompletionBuf *)gpuExec.getExported(
            (uint32_t)ExportID::FlowCompletionBuf);
        FlowCompletionRecord rec {};
        REQ_CUDA(cudaMemcpy(&rec, &dev_buf->records[idx],
                            sizeof(FlowCompletionRecord),
                            cudaMemcpyDeviceToHost));
        return rec;
    }

    inline virtual StepPhaseTimes lastStepPhaseTimes() final
    {
        auto *dev_ptr = (StepPhaseTimes *)gpuExec.getExported(
            (uint32_t)ExportID::StepPhaseTimes);
        StepPhaseTimes host_times {};
        REQ_CUDA(cudaMemcpy(&host_times, dev_ptr, sizeof(StepPhaseTimes),
                            cudaMemcpyDeviceToHost));
        return host_times;
    }
};
#endif

static HeapArray<WorldInit> setupWorldInitData(int64_t num_worlds,
                                               EpisodeManager *episode_mgr,
                                               const GridState *grid,
                                               const NetworkInit *network)
{
    HeapArray<WorldInit> world_inits(num_worlds);

    for (int64_t i = 0; i < num_worlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr,
            grid,
            network,
        };
    }

    return world_inits;
}

Manager::Impl * Manager::Impl::init(const Config &cfg,
                                    const GridState &src_grid,
                                    const NetworkInit &src_network)
{
    static_assert(sizeof(GridState) % alignof(Cell) == 0);

    int32_t perf_fct_only = resolvePerfFCTOnly(cfg);

    Sim::Config sim_cfg {
        .maxEpisodeLength = cfg.maxEpisodeLength,
        .enableViewer = false,
        .default_link_delay = 0.001,
        .propagation_interval = cfg.propagation_interval,
        .enable_buffer = 1,
        .enable_pfc = cfg.enable_pfc,
        .pfc_egress = cfg.pfc_egress,
        .pfc_xoff_threshold = cfg.pfc_xoff_threshold,
        .pfc_xon_threshold = cfg.pfc_xon_threshold,
        .dt_min = cfg.dt_min,
        .qos_mode = cfg.qos_mode,
        .prior_weights = {},
        .perf_fct_only = perf_fct_only,
    };
    for (int i = 0; i < 8; i++) {
        sim_cfg.prior_weights[i] = cfg.prior_weights[i];
    }

    switch (cfg.execMode) {
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };

        uint64_t num_cell_bytes =
            sizeof(Cell) * src_grid.width * src_grid.height;

        auto *grid_data =
            (char *)malloc(sizeof(GridState) + num_cell_bytes);
        Cell *cpu_cell_data = (Cell *)(grid_data + sizeof(GridState));

        GridState *cpu_grid = (GridState *)grid_data;
        *cpu_grid = GridState {
            .cells = cpu_cell_data,
            .startX = src_grid.startX,
            .startY = src_grid.startY,
            .width = src_grid.width,
            .height = src_grid.height,
        };

        memcpy(cpu_cell_data, src_grid.cells, num_cell_bytes);

        NetworkLayout network_layout = getNetworkLayout(src_network);
        auto *network_data = (char *)malloc(network_layout.totalBytes);
        NetworkInit *cpu_network = (NetworkInit *)network_data;
        NodeDef *cpu_nodes = (NodeDef *)(network_data + network_layout.nodesOffset);
        LinkDef *cpu_links = (LinkDef *)(network_data + network_layout.linksOffset);
        FlowDef *cpu_flows = (FlowDef *)(network_data + network_layout.flowsOffset);

        *cpu_network = NetworkInit {
            .nodes = cpu_nodes,
            .numNodes = src_network.numNodes,
            .links = cpu_links,
            .numLinks = src_network.numLinks,
            .flows = cpu_flows,
            .numFlows = src_network.numFlows,
        };

        if (src_network.numNodes > 0) {
            memcpy(cpu_nodes, src_network.nodes,
                   sizeof(NodeDef) * (uint64_t)src_network.numNodes);
        }
        if (src_network.numLinks > 0) {
            memcpy(cpu_links, src_network.links,
                   sizeof(LinkDef) * (uint64_t)src_network.numLinks);
        }
        if (src_network.numFlows > 0) {
            memcpy(cpu_flows, src_network.flows,
                   sizeof(FlowDef) * (uint64_t)src_network.numFlows);
            sortFlowsByStartTime(cpu_flows, src_network.numFlows);
        }

        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            episode_mgr, cpu_grid, cpu_network);

        return new CPUImpl(cfg, sim_cfg, episode_mgr, cpu_grid, cpu_network,
                           world_inits.data());
    } break;
    case ExecMode::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        FATAL("CUDA support not compiled in!");
#else
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(cfg.gpuID);

        EpisodeManager *episode_mgr =
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        uint64_t num_cell_bytes =
            sizeof(Cell) * src_grid.width * src_grid.height;

        auto *grid_data =
            (char *)cu::allocGPU(sizeof(GridState) + num_cell_bytes);

        Cell *gpu_cell_data = (Cell *)(grid_data + sizeof(GridState));
        GridState grid_staging {
            .cells = gpu_cell_data,
            .startX = src_grid.startX,
            .startY = src_grid.startY,
            .width = src_grid.width,
            .height = src_grid.height,
        };

        REQ_CUDA(cudaMemcpy(grid_data, &grid_staging, sizeof(GridState),
                            cudaMemcpyHostToDevice));
        if (num_cell_bytes > 0) {
            REQ_CUDA(cudaMemcpy(gpu_cell_data, src_grid.cells, num_cell_bytes,
                                cudaMemcpyHostToDevice));
        }

        GridState *gpu_grid = (GridState *)grid_data;

        NetworkLayout network_layout = getNetworkLayout(src_network);
        auto *network_data = (char *)cu::allocGPU(network_layout.totalBytes);
        NodeDef *gpu_nodes = (NodeDef *)(network_data + network_layout.nodesOffset);
        LinkDef *gpu_links = (LinkDef *)(network_data + network_layout.linksOffset);
        FlowDef *gpu_flows = (FlowDef *)(network_data + network_layout.flowsOffset);

        NetworkInit network_staging {
            .nodes = gpu_nodes,
            .numNodes = src_network.numNodes,
            .links = gpu_links,
            .numLinks = src_network.numLinks,
            .flows = gpu_flows,
            .numFlows = src_network.numFlows,
        };

        REQ_CUDA(cudaMemcpy(network_data, &network_staging, sizeof(NetworkInit),
                            cudaMemcpyHostToDevice));
        if (src_network.numNodes > 0) {
            REQ_CUDA(cudaMemcpy(gpu_nodes, src_network.nodes,
                                sizeof(NodeDef) * (uint64_t)src_network.numNodes,
                                cudaMemcpyHostToDevice));
        }
        if (src_network.numLinks > 0) {
            REQ_CUDA(cudaMemcpy(gpu_links, src_network.links,
                                sizeof(LinkDef) * (uint64_t)src_network.numLinks,
                                cudaMemcpyHostToDevice));
        }
        if (src_network.numFlows > 0) {
            uint64_t num_flow_bytes =
                sizeof(FlowDef) * (uint64_t)src_network.numFlows;
            HeapArray<FlowDef> sorted_flows(src_network.numFlows);
            memcpy(sorted_flows.data(), src_network.flows, num_flow_bytes);
            sortFlowsByStartTime(sorted_flows.data(), src_network.numFlows);
            REQ_CUDA(cudaMemcpy(gpu_flows, sorted_flows.data(),
                                num_flow_bytes, cudaMemcpyHostToDevice));
        }

        NetworkInit *gpu_network = (NetworkInit *)network_data;

        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            episode_mgr, gpu_grid, gpu_network);

        return new GPUImpl(cu_ctx, cfg, sim_cfg, episode_mgr, gpu_grid,
                           gpu_network, world_inits.data());
#endif
    } break;
    default: return nullptr;
    }
}

Manager::Manager(const Config &cfg,
                 const GridState &src_grid,
                 const NetworkInit &src_network)
    : impl_(Impl::init(cfg, src_grid, src_network))
{}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset, TensorElementType::Int32,
                               {impl_->cfg.numWorlds, 1});
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {impl_->cfg.numWorlds, 1});
}

Tensor Manager::observationTensor() const
{
    return impl_->exportTensor(ExportID::GridPos, TensorElementType::Int32,
        {impl_->cfg.numWorlds, 2});
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
        {impl_->cfg.numWorlds, 1});
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Float32,
        {impl_->cfg.numWorlds, 1});
}

double Manager::simulationTime()
{
    return impl_->simulationTime();
}

int32_t Manager::numFlowDefs()
{
    return impl_->numFlowDefs();
}

int32_t Manager::numPendingFlows()
{
    return impl_->numPendingFlows();
}

int32_t Manager::numDelayedEvents()
{
    return impl_->numDelayedEvents();
}

int32_t Manager::numActiveTags()
{
    return impl_->numActiveTags();
}

int32_t Manager::numSourceTags()
{
    return impl_->numSourceTags();
}

int32_t Manager::numFlowCompletions()
{
    return impl_->numFlowCompletions();
}

FlowCompletionRecord Manager::flowCompletion(int32_t idx)
{
    return impl_->flowCompletion(idx);
}

StepPhaseTimes Manager::lastStepPhaseTimes()
{
    return impl_->lastStepPhaseTimes();
}

}
