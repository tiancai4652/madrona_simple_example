#pragma once
#ifdef gridworld_madrona_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include "grid.hpp"
#include "init.hpp"

namespace madsimple {

class Manager {
public:
    struct Config {
        uint32_t maxEpisodeLength;
        madrona::ExecMode execMode;
        uint32_t numWorlds;
        int gpuID;
        double propagation_interval = 0.0;
        int32_t enable_pfc = 0;
        int32_t pfc_egress = 0;
        double pfc_xoff_threshold = 1e9;
        double pfc_xon_threshold = 0.5e9;
        double dt_min = 0.0;
        int32_t qos_mode = 0;
        double prior_weights[8] = {};
        int32_t perf_fct_only = 1;
        int32_t step_workload = 0;
    };

    MGR_EXPORT Manager(const Config &cfg,
                       const GridState &src_grid,
                       const NetworkInit &src_network);
    MGR_EXPORT ~Manager();

    MGR_EXPORT void step();

    MGR_EXPORT madrona::py::Tensor resetTensor() const;
    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor observationTensor() const;
    MGR_EXPORT madrona::py::Tensor rewardTensor() const;
    MGR_EXPORT madrona::py::Tensor doneTensor() const;

    MGR_EXPORT double simulationTime();
    MGR_EXPORT int32_t numFlowDefs();
    MGR_EXPORT int32_t numPendingFlows();
    MGR_EXPORT int32_t numDelayedEvents();
    MGR_EXPORT int32_t numActiveTags();
    MGR_EXPORT int32_t numSourceTags();
    MGR_EXPORT int32_t numFlowCompletions();
    MGR_EXPORT FlowCompletionRecord flowCompletion(int32_t idx);
    MGR_EXPORT StepWorkloadStats stepWorkload();

private:
    struct Impl;
    struct CPUImpl;
    struct GPUImpl;

    std::unique_ptr<Impl> impl_;
};

}
