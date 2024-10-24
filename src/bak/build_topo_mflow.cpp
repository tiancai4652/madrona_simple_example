#include "sim_mflow.hpp"
// #include "sim_mflow.cpp"

#include <madrona/mw_gpu_entry.hpp> 

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>

#include "types_mflow.hpp"
#include "init.hpp"

#include "fib_nexthop_init_mflow.hpp"
#include "flow_mflow.hpp"

#include <iostream>
#include <vector>
#include <algorithm>

// #include <cstdio>
// #include <cstring>
// #include <cstdlib>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {



template<typename T1, typename T2>
inline bool _dequeue_entt(T1& queue, T2& elem) 
{
    if (queue.cur_num == 0) {
        // The queue is empty, cannot dequeue
        return false;
    }

    elem = queue.flow[queue.head];
    queue.head = (queue.head + 1) % MAX_NPU_FLOW_NUM;
    queue.cur_num -= 1;

    return true; 
}


template<typename T1, typename T2>
inline void _enqueue_entt(T1 &queue, T2 pkt) {
    if (queue.cur_num >= MAX_NPU_FLOW_NUM) {
        printf("error: flow queue is overloaded\n");
        // exit(0);
    }
    queue.flow[queue.tail] = pkt;
    queue.tail = (queue.tail + 1) % MAX_NPU_FLOW_NUM;
    queue.cur_num += 1;
}


// Archtype: Switch
// SwitchType::edge
// SwitchType
// FIBTable
// calc_next_hop()
// const uint32_t k_ary = 8; 
// const uint32_t num_switch = K_ARY*K_ARY*5/4;
// const uint32_t num_host = K_ARY*K_ARY*K_ARY/4;


void generate_switch(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    CountT num_host = k_ary*k_ary*k_ary/4;
    CountT num_next_hop = k_ary;

    printf("*******generate_switch*******\n");
    // build edge switch
    for (uint32_t i = 0; i < num_edge_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        printf("");
        ctx.get<SwitchType>(e_switch) = SwitchType::Edge;
        ctx.get<SwitchID>(e_switch).switch_id = i;

        for (uint32_t j = 0; j < num_host; j++) {
            for (uint32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i][j][k];
            }
        }
        
        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;

        ctx.data()._switches[ctx.data().numSwitch++] = e_switch;
    }

    // build aggragate switch
    for (uint32_t i = 0; i < num_aggr_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        ctx.get<SwitchType>(e_switch) = SwitchType::Aggr;
        ctx.get<SwitchID>(e_switch).switch_id = i+num_edge_switch;

        for (uint32_t j = 0; j < num_host; j++) {
            for (uint32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i+num_edge_switch][j][k];
            }
        }

        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;

        ctx.data()._switches[ctx.data().numSwitch++] = e_switch;
    }

    // build core switch
    for (uint32_t i = 0; i < num_core_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        ctx.get<SwitchType>(e_switch) = SwitchType::Core;
        ctx.get<SwitchID>(e_switch).switch_id = i + num_edge_switch + num_aggr_switch;

        for (uint32_t j = 0; j < num_host; j++) {
            for (uint32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i+num_edge_switch+num_aggr_switch][j][k];
            }
        }

        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;
        
        ctx.data()._switches[ctx.data().numSwitch++] = e_switch;
    }
    //printf("\n");
    printf("Total %d switches, %d in_ports, %d e_ports\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort);
}


void generate_in_port(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    printf("*******Generate in ports*******\n");
    // ingress_ports for edge switch
    ctx.data().numInPort = 0;
    for (uint32_t i = 0; i < num_edge_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = i*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i;
            
            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    // ingress_ports for aggragate switch
    for (uint32_t i = 0; i < num_aggr_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = (num_edge_switch+i)*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i + num_edge_switch;

            // printf("switch_id: %d, local_port_id: %d, global_port_id: %d\n", \
            //         ctx.get<SwitchID>(in_port).switch_id, ctx.get<LocalPortID>(in_port).local_port_id, \
            //         ctx.get<GlobalPortID>(in_port).global_port_id);

            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    // ingress_ports for core switch
    for (uint32_t i = 0; i < num_core_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = (num_edge_switch+num_aggr_switch+i)*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i + num_edge_switch + num_aggr_switch;

            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    printf("Total %d switches, %d in_ports, %d e_ports\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort);
}


void generate_e_port(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    printf("*******Generate egress ports*******\n");
    // egress_ports for edge switch

    ctx.data().numEPort = 0;
    for (uint32_t i = 0; i < num_edge_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j;
            ctx.get<GlobalPortID>(e_port).global_port_id = i*k_ary+j; //i*k_ary+j;

            //pfc
            for (uint32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME; 
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i;

            if(j < k_ary/2) 
                ctx.get<NextHopType>(e_port) = NextHopType::HOST;
            else
                ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            uint32_t ett_idx = ctx.data().numEPort; 
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx]; // next_hop_table also include mapping  switch to host nic

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }

    // egress_ports for aggragate switch
    for (uint32_t i = 0; i < num_aggr_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j; //i*k_ary+j;
            ctx.get<GlobalPortID>(e_port).global_port_id = (i+num_edge_switch)*k_ary+j;

            //pfc
            for (uint32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME; 
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i + num_edge_switch;

            ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            uint32_t ett_idx = ctx.data().numEPort;
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx];

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }

    // egress_ports for core switch
    for (uint32_t i = 0; i < num_core_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {

            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j; //i*k_ary+j;
            ctx.get<GlobalPortID>(e_port).global_port_id = ctx.data().numEPort + (i+num_edge_switch+num_aggr_switch)*k_ary+j;

            //pfc
            for (uint32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME; 
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i + num_edge_switch + num_aggr_switch;
            
            ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            uint32_t ett_idx = ctx.data().numEPort;
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx];

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }
    printf("Total %d switches, %d in_ports, %d e_ports\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort);
}

void generate_host(Engine &ctx, CountT k_ary)
{
    CountT num_host = k_ary*k_ary*k_ary/4;

    printf("*******Generate NPUs*******\n");
    //NPUs
    for (uint32_t i = 0; i < num_host; i++) {
        Entity npu_e = ctx.makeEntity<NPU>();
        ctx.get<NPU_ID>(npu_e).npu_id = i;
        
        ctx.get<SimTime>(npu_e).sim_time = 0;
        ctx.get<SimTimePerUpdate>(npu_e).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

        ctx.get<SendFlows>(npu_e).head = 0;
        ctx.get<SendFlows>(npu_e).tail = 0;
        ctx.get<SendFlows>(npu_e).cur_num = 0;
    }

    printf("*******Generate NICs*******\n");
    //nic
    for (uint32_t i = 0; i < num_host; i++) {
        // if (i > 3) {
        //     break;
        // }
        Entity nic_e = ctx.makeEntity<NIC>();
        ctx.get<NIC_ID>(nic_e).nic_id = i;
        // ctx.get<EgressPortID>(nic_e).egress_port_id = i;

        ctx.get<MountedFlows>(nic_e).head = 0;
        ctx.get<MountedFlows>(nic_e).tail = 0;
        ctx.get<MountedFlows>(nic_e).cur_num = 0;

        ctx.get<BidPktBuf>(nic_e).snd_buf.head = 0;
        ctx.get<BidPktBuf>(nic_e).snd_buf.tail = 0;
        ctx.get<BidPktBuf>(nic_e).snd_buf.cur_num = 0;
        ctx.get<BidPktBuf>(nic_e).snd_buf.cur_bytes = 0;

        ctx.get<BidPktBuf>(nic_e).recv_buf.head = 0;
        ctx.get<BidPktBuf>(nic_e).recv_buf.tail = 0;
        ctx.get<BidPktBuf>(nic_e).recv_buf.cur_num = 0;
        ctx.get<BidPktBuf>(nic_e).recv_buf.cur_bytes = 0;

        ctx.get<TXHistory>(nic_e).head = 0;
        ctx.get<TXHistory>(nic_e).tail = 0;
        ctx.get<TXHistory>(nic_e).cur_num = 0;
        ctx.get<TXHistory>(nic_e).cur_bytes = 0;

        ctx.get<NextHopType>(nic_e) = NextHopType::SWITCH;

        ctx.get<HSLinkDelay>(nic_e).HS_link_delay = HS_LINK_DELAY; // 1 us, 1000 ns 
        ctx.get<NICRate>(nic_e).nic_rate = 1000LL*1000*1000*100; // 100 Gbps 

        uint32_t ett_idx = ctx.data().num_nic;
        ctx.get<NextHop>(nic_e).next_hop = next_hop_table_host_to_sw[ett_idx];

        ctx.get<SimTime>(nic_e).sim_time = 0;

        ctx.get<Seed>(nic_e).seed = i;
        ctx.get<SimTimePerUpdate>(nic_e).sim_time_per_update = LOOKAHEAD_TIME; //1000ns
        
        ctx.data()._nics[ctx.data().num_nic++] = nic_e;
    }

    printf("*******Generate send/recv flows*******\n");
    // send flows 
    for (uint32_t i = 0; i < 2; i++) {
        /////////////////////////////////////////////////////////////////////////////////
        // snd flows
        Entity flow_entt = ctx.makeEntity<SndFlow>();
        ctx.data()._snd_flows[ctx.data().num_snd_flow++] = flow_entt;

        ctx.get<FlowID>(flow_entt).flow_id = flow_events[i][0];
        ctx.get<Src>(flow_entt).src = flow_events[i][1];
        ctx.get<Dst>(flow_entt).dst = flow_events[i][2];
        ctx.get<L4Port>(flow_entt).l4_port = 0; //FIXME
    
        ctx.get<FlowSize>(flow_entt).flow_size = flow_events[i][3];
        ctx.get<StartTime>(flow_entt).start_time = flow_events[i][4];
        ctx.get<StopTime>(flow_entt).stop_time = 0;

        ctx.get<NIC_ID>(flow_entt).nic_id = flow_events[i][5];
        ctx.get<SndServerID>(flow_entt).snd_server_id = flow_events[i][6];
        ctx.get<RecvServerID>(flow_entt).recv_server_id = flow_events[i][7];

        ctx.get<SndNxt>(flow_entt).snd_nxt = 0;
        ctx.get<SndUna>(flow_entt).snd_una = 0;
        ctx.get<FlowState>(flow_entt) = FlowState::UNCOMPLETE;

        ctx.get<LastAckTimestamp>(flow_entt).last_ack_timestamp = 0; //1ms
        ctx.get<NxtPktEvent>(flow_entt).nxt_pkt_event = 0;
        
        ctx.get<PFCState>(flow_entt) = PFCState::RESUME;

        ctx.get<CC_Para>(flow_entt).m_rate = DCQCN_CurrentRate;
        ctx.get<CC_Para>(flow_entt).tar_rate = DCQCN_TargetRate;
        ctx.get<CC_Para>(flow_entt).dcqcn_Alpha = DCQCN_Alpha;
        ctx.get<CC_Para>(flow_entt).dcqcn_G = DCQCN_G;
        ctx.get<CC_Para>(flow_entt).dcqcn_AIRate = DCQCN_AIRate;
        ctx.get<CC_Para>(flow_entt).dcqcn_HAIRate = DCQCN_HAIRate;
        ctx.get<CC_Para>(flow_entt).dcqcn_RateDecreaseInterval = DCQCN_RateDecreaseInterval;
        ctx.get<CC_Para>(flow_entt).dcqcn_RateIncreaseInterval = DCQCN_RateIncreaseInterval;
        ctx.get<CC_Para>(flow_entt).dcqcn_AlphaUpdateInterval = DCQCN_AlphaUpdateInterval;
        ctx.get<CC_Para>(flow_entt).dcqcn_IncreaseStageCount = DCQCN_IncreaseStageCount;
        ctx.get<CC_Para>(flow_entt).dcqcn_RateDecreaseNextTime = DCQCN_RateDecreaseNextTime;
        ctx.get<CC_Para>(flow_entt).dcqcn_RateIncreaseNextTime = DCQCN_RateIncreaseNextTime;
        ctx.get<CC_Para>(flow_entt).dcqcn_AlphaUpdateNextTime = DCQCN_AlphaUpdateNextTime;
        ctx.get<CC_Para>(flow_entt).dcqcn_RateIsDecreased = DCQCN_RateIsDecreased;
        ctx.get<CC_Para>(flow_entt).CNPState = false;

        ctx.get<PktBuf>(flow_entt).head = 0;
        ctx.get<PktBuf>(flow_entt).tail = 0;
        ctx.get<PktBuf>(flow_entt).cur_num = 0;
        ctx.get<PktBuf>(flow_entt).cur_bytes = 0;

        ctx.get<AckPktBuf>(flow_entt).head = 0;
        ctx.get<AckPktBuf>(flow_entt).tail = 0;
        ctx.get<AckPktBuf>(flow_entt).cur_num = 0;
        ctx.get<AckPktBuf>(flow_entt).cur_bytes = 0;

        ctx.get<NICRate>(flow_entt).nic_rate = 1000LL*1000*1000*100; // 100 Gbps  
        ctx.get<HSLinkDelay>(flow_entt).HS_link_delay = HS_LINK_DELAY; // 1 us, 1000 ns 

        ctx.get<SimTime>(flow_entt).sim_time = 0;
        ctx.get<SimTimePerUpdate>(flow_entt).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

        // mounted to NPU
        Entity npu_entt = ctx.data()._npus[ctx.get<Src>(flow_entt).src];
        _enqueue_entt(ctx.get<SendFlows>(npu_entt), flow_entt);

        // mounted to NIC
        Entity nic_entt = ctx.data()._nics[ctx.get<NIC_ID>(flow_entt).nic_id];
        _enqueue_entt(ctx.get<MountedFlows>(nic_entt), flow_entt);
    }

    for (uint32_t i = 0; i < 2; i++) {
        /////////////////////////////////////////////////////////////////////////////////
        // recv flows
        Entity flow_entt = ctx.makeEntity<RecvFlow>();
        ctx.data()._recv_flows[ctx.data().num_recv_flow++] = flow_entt;
        
        ctx.get<FlowID>(flow_entt).flow_id = flow_events[i][0];;
        ctx.get<Src>(flow_entt).src = flow_events[i][1];
        ctx.get<Dst>(flow_entt).dst = flow_events[i][2];
        ctx.get<L4Port>(flow_entt).l4_port = 0;
    
        ctx.get<FlowSize>(flow_entt).flow_size = flow_events[i][3];
        ctx.get<StartTime>(flow_entt).start_time = flow_events[i][4];
        ctx.get<StopTime>(flow_entt).stop_time = 0;

        ctx.get<NIC_ID>(flow_entt).nic_id = flow_events[i][5];
        ctx.get<SndServerID>(flow_entt).snd_server_id = flow_events[i][6];
        ctx.get<RecvServerID>(flow_entt).recv_server_id = flow_events[i][7];

        ctx.get<FlowState>(flow_entt) = FlowState::UNCOMPLETE;

        ctx.get<LastCNPTimestamp>(flow_entt).last_cnp_timestamp = 0; //1ms

        ctx.get<RecvBytes>(flow_entt).recv_bytes = 0;

        ctx.get<PktBuf>(flow_entt).head = 0;
        ctx.get<PktBuf>(flow_entt).tail = 0;
        ctx.get<PktBuf>(flow_entt).cur_num = 0;
        ctx.get<PktBuf>(flow_entt).cur_bytes = 0;

        ctx.get<NICRate>(flow_entt).nic_rate = 1000LL*1000*1000*100; // 100 Gbps  
        ctx.get<HSLinkDelay>(flow_entt).HS_link_delay = HS_LINK_DELAY; // 1 us, 1000 ns 

        ctx.get<SimTime>(flow_entt).sim_time = 0;
        ctx.get<SimTimePerUpdate>(flow_entt).sim_time_per_update = LOOKAHEAD_TIME; //1000ns
    }   

    printf("Total %d switches, %d in_ports, %d e_ports, %d send_flows, %d recv_flows\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, ctx.data().num_snd_flow, ctx.data().num_recv_flow);
}


}