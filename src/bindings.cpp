#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace madsimple {

static void setRewards(Cell *cells,
                       const float *rewards,
                       int64_t grid_x,
                       int64_t grid_y)
{
    for (int64_t y = 0; y < grid_y; y++) {
        for (int64_t x = 0; x < grid_x; x++) {
            int64_t idx = y * grid_x + x;
            cells[idx].reward = rewards[idx];
        }
    }
}

static void tagWalls(Cell *cells,
                     const bool *walls,
                     int64_t grid_x,
                     int64_t grid_y)
{
    for (int64_t y = 0; y < grid_y; y++) {
        for (int64_t x = 0; x < grid_x; x++) {
            int64_t idx = y * grid_x + x;

            if (walls[idx]) {
                cells[idx].flags |= CellFlag::Wall;
            }
        }
    }
}

static void tagEnd(Cell *cells,
                   const int32_t *end_cells,
                   int64_t num_end_cells,
                   int64_t grid_x,
                   int64_t grid_y)
{
    for (int64_t c = 0; c < num_end_cells; c++) {
        int64_t idx = c * 2;
        int64_t y = (int32_t)end_cells[idx];
        int64_t x = (int32_t)end_cells[idx + 1];

        if (x >= grid_x || y >= grid_y) {
            throw std::runtime_error("Out of range end cells");
        }

        cells[y * grid_x + x].flags |= CellFlag::End;
    }
}

static Cell * setupCellData(
    const nb::ndarray<bool, nb::shape<-1, -1>,
        nb::c_contig, nb::device::cpu> &walls,
    const nb::ndarray<float, nb::shape<-1, -1>,
        nb::c_contig, nb::device::cpu> &rewards,
    const nb::ndarray<int32_t, nb::shape<-1, 2>,
        nb::c_contig, nb::device::cpu> &end_cells,
    int64_t grid_x,
    int64_t grid_y)

{
    Cell *cells = new Cell[grid_x * grid_y]();

    setRewards(cells, rewards.data(), grid_x, grid_y);
    tagWalls(cells, walls.data(), grid_x, grid_y);
    tagEnd(cells, end_cells.data(),
        (int64_t)end_cells.shape(0), grid_x, grid_y);

    return cells;
}

static void checkLength(int64_t actual,
                        int64_t expected,
                        const char *name)
{
    if (actual != expected) {
        throw std::runtime_error(std::string(name) + " length mismatch");
    }
}

static NodeType parseNodeType(int32_t raw_type)
{
    switch (raw_type) {
    case 0: return NodeType::Host;
    case 1: return NodeType::Switch;
    default: throw std::runtime_error("Invalid node type value");
    }
}

static NodeDef *setupNodeData(
    const nb::ndarray<int32_t, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &node_ids,
    const nb::ndarray<int32_t, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &node_types,
    const nb::ndarray<double, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &node_port_bws)
{
    int64_t num_nodes = (int64_t)node_ids.shape(0);
    checkLength((int64_t)node_types.shape(0), num_nodes, "node_types");
    checkLength((int64_t)node_port_bws.shape(0), num_nodes, "node_port_bws");

    if (num_nodes > MAX_TOPO_NODES) {
        throw std::runtime_error("Too many nodes for current MAX_TOPO_NODES");
    }

    NodeDef *nodes = new NodeDef[num_nodes > 0 ? num_nodes : 1]();
    for (int64_t i = 0; i < num_nodes; i++) {
        nodes[i] = NodeDef {
            .id = node_ids.data()[i],
            .type = parseNodeType(node_types.data()[i]),
            .port_bw = node_port_bws.data()[i],
        };
    }

    return nodes;
}

static LinkDef *setupLinkData(
    const nb::ndarray<int32_t, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &link_srcs,
    const nb::ndarray<int32_t, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &link_dsts,
    const nb::ndarray<double, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &link_delays,
    const nb::ndarray<double, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &link_bandwidths)
{
    int64_t num_links = (int64_t)link_srcs.shape(0);
    checkLength((int64_t)link_dsts.shape(0), num_links, "link_dsts");
    checkLength((int64_t)link_delays.shape(0), num_links, "link_delays");
    checkLength((int64_t)link_bandwidths.shape(0), num_links, "link_bandwidths");

    if (num_links > MAX_TOPO_LINKS) {
        throw std::runtime_error("Too many links for current MAX_TOPO_LINKS");
    }

    LinkDef *links = new LinkDef[num_links > 0 ? num_links : 1]();
    for (int64_t i = 0; i < num_links; i++) {
        links[i] = LinkDef {
            .src = link_srcs.data()[i],
            .dst = link_dsts.data()[i],
            .delay = link_delays.data()[i],
            .bandwidth = link_bandwidths.data()[i],
        };
    }

    return links;
}

static FlowDef *setupFlowData(
    const nb::ndarray<int64_t, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &flow_ids,
    const nb::ndarray<int32_t, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &flow_src_nodes,
    const nb::ndarray<int32_t, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &flow_dst_nodes,
    const nb::ndarray<double, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &flow_sizes,
    const nb::ndarray<double, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &flow_start_times,
    const nb::ndarray<int32_t, nb::shape<-1>,
        nb::c_contig, nb::device::cpu> &flow_priorities)
{
    int64_t num_flows = (int64_t)flow_ids.shape(0);
    checkLength((int64_t)flow_src_nodes.shape(0), num_flows, "flow_src_nodes");
    checkLength((int64_t)flow_dst_nodes.shape(0), num_flows, "flow_dst_nodes");
    checkLength((int64_t)flow_sizes.shape(0), num_flows, "flow_sizes");
    checkLength((int64_t)flow_start_times.shape(0), num_flows, "flow_start_times");
    checkLength((int64_t)flow_priorities.shape(0), num_flows, "flow_priorities");

    if (num_flows > MAX_FLOWS) {
        throw std::runtime_error("Too many flows for current MAX_FLOWS");
    }

    FlowDef *flows = new FlowDef[num_flows > 0 ? num_flows : 1]();
    for (int64_t i = 0; i < num_flows; i++) {
        flows[i] = FlowDef {
            .id = flow_ids.data()[i],
            .src_node = flow_src_nodes.data()[i],
            .dst_node = flow_dst_nodes.data()[i],
            .size = flow_sizes.data()[i],
            .start_time = flow_start_times.data()[i],
            .priority = flow_priorities.data()[i],
        };
    }

    return flows;
}

NB_MODULE(_madrona_simple_example_cpp, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "SimpleGridworldSimulator")
        .def("__init__", [](Manager *self,
                            nb::ndarray<bool, nb::shape<-1, -1>,
                                nb::c_contig, nb::device::cpu> walls,
                            nb::ndarray<float, nb::shape<-1, -1>,
                                nb::c_contig, nb::device::cpu> rewards,
                            nb::ndarray<int32_t, nb::shape<-1, 2>,
                                nb::c_contig, nb::device::cpu> end_cells,
                            int64_t start_x,
                            int64_t start_y,
                            nb::ndarray<int32_t, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> node_ids,
                            nb::ndarray<int32_t, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> node_types,
                            nb::ndarray<double, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> node_port_bws,
                            nb::ndarray<int32_t, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> link_srcs,
                            nb::ndarray<int32_t, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> link_dsts,
                            nb::ndarray<double, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> link_delays,
                            nb::ndarray<double, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> link_bandwidths,
                            nb::ndarray<int64_t, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> flow_ids,
                            nb::ndarray<int32_t, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> flow_src_nodes,
                            nb::ndarray<int32_t, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> flow_dst_nodes,
                            nb::ndarray<double, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> flow_sizes,
                            nb::ndarray<double, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> flow_start_times,
                            nb::ndarray<int32_t, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> flow_priorities,
                            int64_t max_episode_length,
                            madrona::py::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t gpu_id,
                            double propagation_interval,
                            int32_t enable_pfc,
                            int32_t pfc_egress,
                            double pfc_xoff_threshold,
                            double pfc_xon_threshold,
                            double dt_min,
                            int32_t qos_mode,
                            nb::ndarray<double, nb::shape<-1>,
                                nb::c_contig, nb::device::cpu> prior_weights) {
            int64_t grid_y = (int64_t)walls.shape(0);
            int64_t grid_x = (int64_t)walls.shape(1);

            if ((int64_t)rewards.shape(0) != grid_y ||
                (int64_t)rewards.shape(1) != grid_x) {
                throw std::runtime_error("walls and rewards shapes don't match");
            }

            std::unique_ptr<Cell[]> cells(
                setupCellData(walls, rewards, end_cells, grid_x, grid_y));
            std::unique_ptr<NodeDef[]> nodes(
                setupNodeData(node_ids, node_types, node_port_bws));
            std::unique_ptr<LinkDef[]> links(
                setupLinkData(link_srcs, link_dsts, link_delays, link_bandwidths));
            std::unique_ptr<FlowDef[]> flows(
                setupFlowData(flow_ids, flow_src_nodes, flow_dst_nodes,
                              flow_sizes, flow_start_times, flow_priorities));

            double prior_weights_arr[8] = {};
            int64_t num_weights = (int64_t)prior_weights.shape(0);
            for (int64_t i = 0; i < num_weights && i < 8; i++) {
                prior_weights_arr[i] = prior_weights.data()[i];
            }

            new (self) Manager(Manager::Config {
                .maxEpisodeLength = (uint32_t)max_episode_length,
                .execMode = exec_mode,
                .numWorlds = (uint32_t)num_worlds,
                .gpuID = (int)gpu_id,
                .propagation_interval = propagation_interval,
                .enable_pfc = enable_pfc,
                .pfc_egress = pfc_egress,
                .pfc_xoff_threshold = pfc_xoff_threshold,
                .pfc_xon_threshold = pfc_xon_threshold,
                .dt_min = dt_min,
                .qos_mode = qos_mode,
                .prior_weights = {prior_weights_arr[0], prior_weights_arr[1],
                                  prior_weights_arr[2], prior_weights_arr[3],
                                  prior_weights_arr[4], prior_weights_arr[5],
                                  prior_weights_arr[6], prior_weights_arr[7]},
            }, GridState {
                .cells = cells.get(),
                .startX = (int32_t)start_x,
                .startY = (int32_t)start_y,
                .width = (int32_t)grid_x,
                .height = (int32_t)grid_y,
            }, NetworkInit {
                .nodes = nodes.get(),
                .numNodes = (int32_t)node_ids.shape(0),
                .links = links.get(),
                .numLinks = (int32_t)link_srcs.shape(0),
                .flows = flows.get(),
                .numFlows = (int32_t)flow_ids.shape(0),
            });
        }, nb::arg("walls"),
           nb::arg("rewards"),
           nb::arg("end_cells"),
           nb::arg("start_x"),
           nb::arg("start_y"),
           nb::arg("node_ids"),
           nb::arg("node_types"),
           nb::arg("node_port_bws"),
           nb::arg("link_srcs"),
           nb::arg("link_dsts"),
           nb::arg("link_delays"),
           nb::arg("link_bandwidths"),
           nb::arg("flow_ids"),
           nb::arg("flow_src_nodes"),
           nb::arg("flow_dst_nodes"),
           nb::arg("flow_sizes"),
           nb::arg("flow_start_times"),
           nb::arg("flow_priorities"),
           nb::arg("max_episode_length"),
           nb::arg("exec_mode"),
           nb::arg("num_worlds"),
           nb::arg("gpu_id") = -1,
           nb::arg("propagation_interval") = 0.0,
           nb::arg("enable_pfc") = 0,
           nb::arg("pfc_egress") = 0,
           nb::arg("pfc_xoff_threshold") = 1e9,
           nb::arg("pfc_xon_threshold") = 0.5e9,
           nb::arg("dt_min") = 0.0,
           nb::arg("qos_mode") = 0,
           nb::arg("prior_weights") = nb::ndarray<double, nb::shape<-1>, nb::c_contig, nb::device::cpu>())
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("simulation_time", &Manager::simulationTime)
        .def("num_flow_defs", &Manager::numFlowDefs)
        .def("num_pending_flows", &Manager::numPendingFlows)
        .def("num_delayed_events", &Manager::numDelayedEvents)
        .def("num_active_tags", &Manager::numActiveTags)
        .def("num_source_tags", &Manager::numSourceTags)
        .def("num_flow_completions", &Manager::numFlowCompletions)
        .def("last_step_phase_times", [](Manager &mgr) {
            StepPhaseTimes times = mgr.lastStepPhaseTimes();
            nb::dict result;
            result["step"] = nb::int_(times.step);
            result["total_wall_time_s"] = nb::float_(times.totalWallTimeS);
            result["schedule_wall_time_s"] =
                nb::float_(times.phaseWallTimeS[(uint32_t)StepPhaseID::Schedule]);
            result["deliver_wall_time_s"] =
                nb::float_(times.phaseWallTimeS[(uint32_t)StepPhaseID::Deliver]);
            result["ingress_wall_time_s"] =
                nb::float_(times.phaseWallTimeS[(uint32_t)StepPhaseID::Ingress]);
            result["alloc_wall_time_s"] =
                nb::float_(times.phaseWallTimeS[(uint32_t)StepPhaseID::Alloc]);
            result["pfc_emit_wall_time_s"] =
                nb::float_(times.phaseWallTimeS[(uint32_t)StepPhaseID::PfcEmit]);
            result["clear_dt_wall_time_s"] =
                nb::float_(times.phaseWallTimeS[(uint32_t)StepPhaseID::ClearDT]);
            result["buffer_progress_wall_time_s"] =
                nb::float_(times.phaseWallTimeS[(uint32_t)StepPhaseID::BufferProgress]);
            return result;
        })
        .def("flow_completion", [](Manager &mgr, int64_t idx) {
            FlowCompletionRecord record = mgr.flowCompletion((int32_t)idx);
            nb::dict result;
            result["flow_id"] = nb::int_(record.flow_id);
            result["src_node"] = nb::int_(record.src_node);
            result["dst_node"] = nb::int_(record.dst_node);
            result["size_bytes"] = nb::float_(record.size);
            result["start_time_ms"] = nb::float_(record.start_time);
            result["end_time_ms"] = nb::float_(record.end_time);
            result["fct_ms"] = nb::float_(record.fct());
            result["priority"] = nb::int_(record.priority);
            return result;
        })
    ;
}

}
