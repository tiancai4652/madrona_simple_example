#include "sim.hpp"

#include <madrona/mw_gpu_entry.hpp> 

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>

#include "types.hpp"
#include "init.hpp"

#include "fib_nexthop_init.hpp"

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
// const int32_t num_switch = K_ARY*K_ARY*5/4;
// const int32_t num_host = K_ARY*K_ARY*K_ARY/4;


void generate_switch(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    CountT num_host = k_ary*k_ary*k_ary/4;
    CountT num_next_hop = k_ary;

    printf("*******generate_switch*******\n");
    // build edge switch
    for (int32_t i = 0; i < num_edge_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        ctx.get<SwitchType>(e_switch) = SwitchType::Edge;
        ctx.get<SwitchID>(e_switch).switch_id = i;

        for (int32_t j = 0; j < num_host; j++) {
            for (int32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i][j][k];
            }
        }
        
        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;

        ctx.data()._switch[ctx.data().numSwitch++] = e_switch;
    }

    // build aggragate switch
    for (int32_t i = 0; i < num_aggr_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        ctx.get<SwitchType>(e_switch) = SwitchType::Aggr;
        ctx.get<SwitchID>(e_switch).switch_id = i+num_edge_switch;

        for (int32_t j = 0; j < num_host; j++) {
            for (int32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i+num_edge_switch][j][k];
            }
        }

        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;

        ctx.data()._switch[ctx.data().numSwitch++] = e_switch;
    }

    // build core switch
    for (int32_t i = 0; i < num_core_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        ctx.get<SwitchType>(e_switch) = SwitchType::Core;
        ctx.get<SwitchID>(e_switch).switch_id = i + num_edge_switch + num_aggr_switch;

        for (int32_t j = 0; j < num_host; j++) {
            for (int32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i+num_edge_switch+num_aggr_switch][j][k];
            }
        }

        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;
        
        ctx.data()._switch[ctx.data().numSwitch++] = e_switch;
    }
    //printf("\n");
    printf("Total %d switches, %d in_ports, %d e_ports, %d hosts\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, ctx.data().numSender);
}


void generate_in_port(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    printf("*******Generate in ports*******\n");
    // ingress_ports for edge switch
    ctx.data().numInPort = 0;
    for (int32_t i = 0; i < num_edge_switch; i++) {
        for (int32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = i*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_pkt_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i;
            
            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEADT_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    // ingress_ports for aggragate switch
    for (int32_t i = 0; i < num_aggr_switch; i++) {
        for (int32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = (num_edge_switch+i)*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_pkt_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i + num_edge_switch;

            // printf("switch_id: %d, local_port_id: %d, global_port_id: %d\n", \
            //         ctx.get<SwitchID>(in_port).switch_id, ctx.get<LocalPortID>(in_port).local_port_id, \
            //         ctx.get<GlobalPortID>(in_port).global_port_id);

            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEADT_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    // ingress_ports for core switch
    for (int32_t i = 0; i < num_core_switch; i++) {
        for (int32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = (num_edge_switch+num_aggr_switch+i)*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_pkt_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i + num_edge_switch + num_aggr_switch;

            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEADT_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    printf("Total %d switches, %d in_ports, %d e_ports, %d hosts\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, ctx.data().numSender);
}


void generate_e_port(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    printf("*******Generate egress ports*******\n");
    // egress_ports for edge switch

    ctx.data().numEPort = 0;
    for (int32_t i = 0; i < num_edge_switch; i++) {
        for (int32_t j = 0; j < k_ary; j++) {
            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j;
            ctx.get<GlobalPortID>(e_port).global_port_id = i*k_ary+j; //i*k_ary+j;

            //pfc
            for (int32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_pkt_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME; 
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_pkt_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i;

            if(j < k_ary/2) 
                ctx.get<NextHopType>(e_port) = NextHopType::HOST;
            else
                ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            int32_t ett_idx = ctx.data().numEPort; 
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx]; // next_hop_table also include mapping  switch to host nic

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEADT_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }

    // egress_ports for aggragate switch
    for (int32_t i = 0; i < num_aggr_switch; i++) {
        for (int32_t j = 0; j < k_ary; j++) {
            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j; //i*k_ary+j;
            ctx.get<GlobalPortID>(e_port).global_port_id = (i+num_edge_switch)*k_ary+j;

            //pfc
            for (int32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_pkt_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME; 
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_pkt_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i + num_edge_switch;

            ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            int32_t ett_idx = ctx.data().numEPort;
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx];

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEADT_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }

    // egress_ports for core switch
    for (int32_t i = 0; i < num_core_switch; i++) {
        for (int32_t j = 0; j < k_ary; j++) {

            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j; //i*k_ary+j;
            ctx.get<GlobalPortID>(e_port).global_port_id = ctx.data().numEPort + (i+num_edge_switch+num_aggr_switch)*k_ary+j;

            //pfc
            for (int32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_pkt_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME; 
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_pkt_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i + num_edge_switch + num_aggr_switch;
            
            ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            int32_t ett_idx = ctx.data().numEPort;
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx];

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEADT_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }
    printf("Total %d switches, %d in_ports, %d e_ports, %d hosts\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, ctx.data().numSender);
}


void generate_host(Engine &ctx, CountT k_ary)
{
    CountT num_host = k_ary*k_ary*k_ary/4;

    printf("*******Generate sender and receiver*******\n");
    //sender
    for (uint32_t i = 0; i < num_host; i++) {
        Entity sender_e = ctx.makeEntity<Sender>();
        ctx.get<SenderID>(sender_e).sender_id = i;
        // ctx.get<EgressPortID>(sender_e).egress_port_id = i;

        ctx.get<PktBuf>(sender_e).head = 0;
        ctx.get<PktBuf>(sender_e).tail = 0;
        ctx.get<PktBuf>(sender_e).cur_pkt_num = 0;
        ctx.get<PktBuf>(sender_e).cur_bytes = 0;

        ctx.get<NextHopType>(sender_e) = NextHopType::SWITCH;

        ctx.get<HSLinkDelay>(sender_e).HS_link_delay = HS_LINK_DELAY; // 1 us, 1000 ns 
        ctx.get<NICRate>(sender_e).nic_rate = 1000LL*1000*1000*100; // 100 Gbps  

        int32_t ett_idx = ctx.data().numSender;
        ctx.get<NextHop>(sender_e).next_hop = next_hop_table_host_to_sw[ett_idx];

        ctx.get<SimTime>(sender_e).sim_time = 0;
        ctx.get<SimTimePerUpdate>(sender_e).sim_time_per_update = LOOKAHEADT_TIME; //1000ns
        
        ctx.data().senders[ctx.data().numSender++] = sender_e;

        ctx.get<SendFlowQueue>(sender_e).head = 0;
        ctx.get<SendFlowQueue>(sender_e).tail = 0;

        // ctx.get<SendFlowQueue>(sender_e).send_flow[0] = {i, i+2, 0, 0, flow_size, 0, 0, 0, FlowState::UNCOMPLETE, 1000LL*1000*1000*100, 1000LL*1000*1000*100};
        int64_t flow_size = 512*1460; // 20000;
        // if (i != 0) {
        //     flow_size = 0; //512*1460;
        // }
        if (i >= num_host/2) {
            flow_size = 0; //512*1460;
        }
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].src = i;
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].dst = i+num_host/2; //i+1; 
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].flow_size = flow_size;
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].sq_num = 0;
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].snd_nxt = 0;
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].snd_una = 0;
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].flow_state = FlowState::UNCOMPLETE;
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].cur_rate = 1000LL*1000*1000*100;
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].tar_rate = 1000LL*1000*1000*100;
        ctx.get<SendFlowQueue>(sender_e).send_flow[0].nxt_pkt_event = 0;

        //printf("src is %d\n", ctx.get<SendFlowQueue>(sender_e).send_flow[0].src);
    }

    //receiver
    for (uint32_t i = 0; i < num_host; i++) {
        // if (i > 1) {
        //     continue;
        // }

        Entity receiver_e = ctx.makeEntity<Receiver>();
        ctx.get<ReceiverID>(receiver_e).receiver_id = i;
        // ctx.get<IngressPortID>(receiver_e).ingress_port_id = i;

        ctx.get<PktBuf>(receiver_e).head = 0;
        ctx.get<PktBuf>(receiver_e).tail = 0;
        ctx.get<PktBuf>(receiver_e).cur_pkt_num = 0;
        ctx.get<PktBuf>(receiver_e).cur_bytes = 0;

        printf("receiver_id: %d, pkt.head: %d, pkt.tail: %d\n", \
                i, ctx.get<PktBuf>(receiver_e).head, ctx.get<PktBuf>(receiver_e).tail);

        ctx.get<NextHopType>(receiver_e) = NextHopType::SWITCH;

        ctx.get<HSLinkDelay>(receiver_e).HS_link_delay = HS_LINK_DELAY; // 1 us, 1000 ns 
        ctx.get<NICRate>(receiver_e).nic_rate = 1000LL*1000*1000*100; // 100 Gbps

        int32_t ett_idx = ctx.data().numReceiver;
        ctx.get<NextHop>(receiver_e).next_hop = next_hop_table_host_to_sw[ett_idx];

        ctx.get<SimTime>(receiver_e).sim_time = 0;
        ctx.get<SimTimePerUpdate>(receiver_e).sim_time_per_update = LOOKAHEADT_TIME; //1000ns

        ctx.data().receivers[ctx.data().numReceiver++] = receiver_e;

        ctx.get<RecvFlowQueue>(receiver_e).head = 0;
        ctx.get<RecvFlowQueue>(receiver_e).tail = 0;

        // if (i != 2) {
        //     continue;
        // }
        // if (i < num_host/2) {
        //     continue;
        // }
        // ctx.get<RecvFlowQueue>(receiver_e).recv_flow[0] = {i-2, i, 0, 0, FlowState::UNCOMPLETE};
        ctx.get<RecvFlowQueue>(receiver_e).recv_flow[0].src = i-num_host/2;  //i-1;
        ctx.get<RecvFlowQueue>(receiver_e).recv_flow[0].dst = i;
        ctx.get<RecvFlowQueue>(receiver_e).recv_flow[0].recv_bytes = 0;
        ctx.get<RecvFlowQueue>(receiver_e).recv_flow[0].flow_state = FlowState::UNCOMPLETE;  
    }
    printf("Total %d switches, %d in_ports, %d e_ports, %d sender, %d receiver\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, ctx.data().numSender, ctx.data().numReceiver);
}


}