/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/Workload.hh"

#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/MemEventHandlerData.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include <json/json.hpp>

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>

using namespace std;
using namespace AstraSim;
using namespace Chakra;
using json = nlohmann::json;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

CUDA_HOST_DEVICE
Workload::Workload(Sys* sys, const char* et_filename, const char* comm_group_filename) {
    this->sys = sys;
    this->et_feeder = nullptr;  // 在GPU上不支持文件操作
    this->comm_group = nullptr;
    this->hw_resource = new HardwareResource(1);  // 默认1个NPU
    this->is_finished = false;
    
    // 默认配置
    this->config.simulation_speed = 1.0f;
    this->config.accuracy = 0.95f;
    this->config.max_iterations = 1000;
    this->config.trace_enabled = false;
    this->config.replay_only = false;
    
    // 初始化通信组
    initialize_comm_group(comm_group_filename);
}

CUDA_HOST_DEVICE
Workload::~Workload() {
    if (comm_group != nullptr) {
        delete comm_group;
    }
    if (et_feeder != nullptr) {
        delete et_feeder;
    }
    if (hw_resource != nullptr) {
        delete hw_resource;
    }
}

CUDA_HOST_DEVICE
void Workload::initialize_comm_group(const char* comm_group_filename) {
    // GPU上简化的通信组初始化
    comm_group = nullptr;
}

CUDA_HOST_DEVICE
void Workload::issue_dep_free_nodes() {
    if (!et_feeder) return;
    
    std::queue<shared_ptr<Chakra::ETFeederNode>> push_back_queue;
    shared_ptr<Chakra::ETFeederNode> node = et_feeder->getNextIssuableNode();
    
    while (node != nullptr) {
        if (hw_resource->is_available(node)) {
            issue(node);
        } else {
            push_back_queue.push(node);
        }
        node = et_feeder->getNextIssuableNode();
    }

    while (!push_back_queue.empty()) {
        shared_ptr<Chakra::ETFeederNode> node = push_back_queue.front();
        et_feeder->pushBackIssuableNode(node->id());
        push_back_queue.pop();
    }
}

CUDA_HOST_DEVICE
void Workload::issue(shared_ptr<Chakra::ETFeederNode> node) {
    if (config.replay_only) {
        hw_resource->occupy(node);
        issue_replay(node);
        return;
    }

    switch (node->type()) {
        case ChakraNodeType::MEM_LOAD_NODE:
        case ChakraNodeType::MEM_STORE_NODE:
            issue_remote_mem(node);
            break;
            
        case ChakraNodeType::COMP_NODE:
            if (node->runtime() == 0 && node->num_ops() == 0) {
                skip_invalid(node);
            } else {
                issue_comp(node);
            }
            break;
            
        case ChakraNodeType::COMM_COLL_NODE:
        case ChakraNodeType::COMM_SEND_NODE:
        case ChakraNodeType::COMM_RECV_NODE:
            issue_comm(node);
            break;
            
        case ChakraNodeType::INVALID_NODE:
            skip_invalid(node);
            break;
    }
}

CUDA_HOST_DEVICE
void Workload::issue_replay(shared_ptr<Chakra::ETFeederNode> node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node->id();
    uint64_t runtime = node->runtime() > 0 ? node->runtime() * 1000 : 1;
    sys->register_event(this, EventType::General, wlhd, runtime);
}

CUDA_HOST_DEVICE
void Workload::issue_remote_mem(shared_ptr<Chakra::ETFeederNode> node) {
    hw_resource->occupy(node);
    
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node->id();
    sys->remote_mem->issue(node->tensor_size(), wlhd);
}

CUDA_HOST_DEVICE
void Workload::issue_comp(shared_ptr<Chakra::ETFeederNode> node) {
    hw_resource->occupy(node);

    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node->id();

    if (sys->roofline_enabled) {
        double operational_intensity = static_cast<double>(node->num_ops()) / static_cast<double>(node->tensor_size());
        double perf = sys->roofline->get_perf(operational_intensity);
        double elapsed_time = static_cast<double>(node->num_ops()) / perf;
        uint64_t runtime = static_cast<uint64_t>(elapsed_time);
        sys->register_event(this, EventType::General, wlhd, runtime);
    } else {
        issue_replay(node);
    }
}

CUDA_HOST_DEVICE
void Workload::issue_comm(shared_ptr<Chakra::ETFeederNode> node) {
    hw_resource->occupy(node);

    // 简化的通信处理
    if (node->type() == ChakraNodeType::COMM_COLL_NODE) {
        DataSet* dataset = nullptr;
        
        switch (node->comm_type()) {
            case ChakraCollectiveCommType::ALL_REDUCE:
                dataset = sys->generate_all_reduce(node->comm_size(), {true}, comm_group, node->comm_priority());
                break;
                
            case ChakraCollectiveCommType::ALL_TO_ALL:
                dataset = sys->generate_all_to_all(node->comm_size(), {true}, comm_group, node->comm_priority());
                break;
                
            case ChakraCollectiveCommType::ALL_GATHER:
                dataset = sys->generate_all_gather(node->comm_size(), {true}, comm_group, node->comm_priority());
                break;
                
            case ChakraCollectiveCommType::REDUCE_SCATTER:
                dataset = sys->generate_reduce_scatter(node->comm_size(), {true}, comm_group, node->comm_priority());
                break;
                
            default:
                // 不支持的集合通信类型
                break;
        }
        
        if (dataset) {
            collective_comm_node_id_map[dataset->my_id] = node->id();
            collective_comm_wrapper_map[dataset->my_id] = dataset;
            dataset->set_notifier(this, EventType::CollectiveCommunicationFinished);
        }
    }
}

CUDA_HOST_DEVICE
void Workload::skip_invalid(shared_ptr<Chakra::ETFeederNode> node) {
    if (et_feeder) {
        et_feeder->freeChildrenNodes(node->id());
        et_feeder->removeNode(node->id());
    }
}

CUDA_HOST_DEVICE
void Workload::call(EventType event, CallData* data) {
    if (is_finished) return;

    if (event == EventType::CollectiveCommunicationFinished) {
        IntData* int_data = (IntData*)data;
        uint64_t node_id = collective_comm_node_id_map[int_data->data];
        
        if (et_feeder) {
            shared_ptr<Chakra::ETFeederNode> node = et_feeder->lookupNode(node_id);
            hw_resource->release(node);
            et_feeder->freeChildrenNodes(node_id);
            et_feeder->removeNode(node_id);
        }
        
        delete collective_comm_wrapper_map[int_data->data];
        collective_comm_wrapper_map.erase(int_data->data);
        
        issue_dep_free_nodes();
    } else {
        if (data == nullptr) {
            issue_dep_free_nodes();
        } else {
            WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
            
            if (et_feeder) {
                shared_ptr<Chakra::ETFeederNode> node = et_feeder->lookupNode(wlhd->node_id);
                hw_resource->release(node);
                et_feeder->freeChildrenNodes(node->id());
                et_feeder->removeNode(wlhd->node_id);
            }
            
            delete wlhd;
            issue_dep_free_nodes();
        }
    }

    // 检查是否完成
    if (!et_feeder || (!et_feeder->hasNodesToIssue() && 
        hw_resource->num_in_flight_cpu_ops == 0 &&
        hw_resource->num_in_flight_gpu_comp_ops == 0 && 
        hw_resource->num_in_flight_gpu_comm_ops == 0)) {
        report();
        is_finished = true;
    }
}

CUDA_HOST_DEVICE
void Workload::fire() {
    call(EventType::General, nullptr);
}

CUDA_HOST_DEVICE
void Workload::report() {
    // GPU上简化的报告
    is_finished = true;
}
