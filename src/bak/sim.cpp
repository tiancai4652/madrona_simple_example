#include "sim.hpp"
#include "build_topo.hpp"
#include "myQueue.hpp"

#include <madrona/mw_gpu_entry.hpp> 
#include <atomic>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();

    registry.registerArchetype<Agent>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);


    //***************************************************
    // for the router forwarding function, of GPU_acclerated DES
    registry.registerComponent<NICRate>();
    registry.registerComponent<HSLinkDelay>();

    registry.registerComponent<NextHopType>();
    registry.registerComponent<NextHop>();
    
    registry.registerComponent<Pkt>();
    registry.registerComponent<PktBuf>();

    registry.registerComponent<SimTime>();
    registry.registerComponent<SimTimePerUpdate>();

    registry.registerComponent<SenderID>();
    registry.registerComponent<EgressPortID>();
    registry.registerArchetype<Sender>();

    registry.registerComponent<ReceiverID>();
    registry.registerComponent<IngressPortID>();
    registry.registerArchetype<Receiver>();

    registry.registerComponent<SwitchID>();
    registry.registerComponent<SwitchType>();
    registry.registerComponent<FIBTable>();
    registry.registerArchetype<Switch>();

    registry.registerComponent<LinkRate>();
    registry.registerComponent<SSLinkDelay>();

    registry.registerComponent<SchedTrajType>();
    registry.registerComponent<PortType>();
    registry.registerComponent<PortID>();

    registry.registerComponent<TXHistory>();  

    registry.registerComponent<ForwardPlan>();

    registry.registerArchetype<IngressPort>();
    registry.registerArchetype<EgressPort>();
}


// inline bool fetch_pkt(PktBuf& queue, int32_t idx, Pkt& packet) 
// {
//     if (queue.head == queue.tail) {
//         // The queue is empty, cannot dequeue
//         return false;
//     }

//     packet = queue.pkts[(queue.head+idx) % PKT_BUF_LEN];
//     //queue.head = (queue.head + 1) % (PKT_BUF_LEN); // use modulo operation to wrap around

//     return true; 
// }


// inline void _enqueue(PktBuf &queue, Pkt packet) {
//     //queue.pkts[queue.tail] = {packet.src_id, packet.dest_id, packet.pkt_size, packet.flow_id, packet.enqueue_time, packet.dequeue_time};
//     queue.pkts[queue.tail] = packet;
//     queue.tail = (queue.tail + 1) % (PKT_BUF_LEN); // consider the extra 10 spaces
// }

// inline int16_t get_queue_len(PktBuf &queue) {
//     if (queue.tail >= queue.head) return queue.tail-queue.head;
//     else return (queue.tail-queue.head)+PKT_BUF_LEN;
// }

template<typename T, size_t N>
size_t getArrayLength(T (&)[N]) {
    return N;
}

template<typename T>
inline Pkt& get_pkt(T& queue, int32_t idx) 
{
    return queue.pkts[(queue.head+idx) % getArrayLength(queue.pkts)];
}

template<typename T1, typename T2>
inline bool _dequeue(T1& queue, T2& packet) 
{
    if (queue.head == queue.tail) {
        // The queue is empty, cannot dequeue
        return false;
    }

    packet = queue.pkts[queue.head];
    queue.head = (queue.head + 1) % (getArrayLength(queue.pkts)); // use modulo operation to wrap around

    return true; 
}

template<typename T1, typename T2>
inline bool fetch_pkt(T1& queue, int32_t idx, T2& elem) 
{
    if (queue.head == queue.tail) {
        // The queue is empty, cannot dequeue
        return false;
    }

    elem = queue.pkts[(queue.head+idx) % getArrayLength(queue.pkts)];
    //queue.head = (queue.head + 1) % (PKT_BUF_LEN); // use modulo operation to wrap around

    return true; 
}

template<typename T1, typename T2>
inline void _enqueue(T1 &queue, T2 packet) {
    //queue.pkts[queue.tail] = {packet.src_id, packet.dest_id, packet.pkt_size, packet.flow_id, packet.enqueue_time, packet.dequeue_time};
    queue.pkts[queue.tail] = packet;
    queue.tail = (queue.tail + 1) % getArrayLength(queue.pkts); // consider the extra 10 spaces
}

template<typename T>
inline int16_t get_queue_len(T &queue) {
    if (queue.tail >= queue.head) return queue.tail-queue.head;
    else return (queue.tail-queue.head)+getArrayLength(queue.pkts);
}

template<typename T>
inline void clear_queue(T &queue) {
    queue.head = 0;
    queue.tail = 0;
}


// create new packets and move them into direct-connected ingress ports
inline void send(Engine &ctx, SenderID &_sender_id, NextHop &_next_hop, \
                 NICRate &_nic_rate, HSLinkDelay &_HS_link_delay, SimTime &_sim_time, \
                 SimTimePerUpdate &_sim_time_per_update) {
    // udp sending
    if (_sender_id.sender_id == 0) {
        printf("*******Enter into send*********\n");
    }
    Entity in_port = ctx.data().inPorts[_next_hop.next_hop];
    //printf("send new packets into the buffer of directed-connected in_port\n");
    int64_t end_time = _sim_time.sim_time + _HS_link_delay.HS_link_delay;
    //printf("_HS_link_delay.HS_link_delay: %lld\n", _HS_link_delay.HS_link_delay);
    //printf("_nic_rate.nic_rate: %lld\n", _nic_rate.nic_rate);
    int16_t pkt_size = 1250; //bytes
    int16_t flow_id = 0;
    int16_t count_pkt = 0;
    int32_t receiver_id = _sender_id.sender_id+ctx.data().numReceiver/2;
    while (_sim_time.sim_time < end_time) {
        //printf("_sim_time.sim_time: %lld, end_time: %lld\n", _sim_time.sim_time, end_time);
        int64_t dequeue_time = _sim_time.sim_time + (pkt_size*8*1.0/_nic_rate.nic_rate)*(1000*1000*1000); // unit is ns
        //printf("dequeue_time: %lld\n", dequeue_time);
        int64_t enqueue_time = dequeue_time + _HS_link_delay.HS_link_delay;

        Pkt packet = {_sender_id.sender_id, receiver_id, pkt_size, flow_id, enqueue_time, dequeue_time};
        _enqueue(ctx.get<PktBuf>(in_port), packet);
        count_pkt += 1;
        _sim_time.sim_time = dequeue_time;
    }


    // int32_t pkt_num = _HS_link_delay.HS_link_delay/PACKET_PROCESS_TIME;
    // if (_sender_id.sender_id < ctx.data().numSender/2){
    //     for (int32_t idx = 0; idx < pkt_num; idx++) {
    //         int64_t enqueue_time = ((idx+1)%(pkt_num-3))*3;
    //         // int64_t enqueue_time = _sim_time.sim_time+PACKET_PROCESS_TIME;
    //         int32_t receiver_id = _sender_id.sender_id+ctx.data().numReceiver/2;   //  communication pair: (0-->5, 1-->6, 2-->7,...,4-->9)
    //         int16_t pkt_size = 1500;
    //         int16_t flow_id = 0;
    //         Pkt packet = {_sender_id.sender_id, receiver_id, pkt_size, flow_id, enqueue_time, 0};
    //         _enqueue(ctx.get<PktBuf>(in_port), packet);
    //     }
    // }
    
    //printf("pkt_buf.tail, %d; pkt_buf.head, %d\n", ctx.get<PktBuf>(in_port).tail, ctx.get<PktBuf>(in_port).head);
    // for (CountT idx = 0; idx < pkt_num; idx++){
    //     printf("%d, ", ctx.get<PktBuf>(in_port).pkts[idx].enqueue_time);
    // }
    //printf("\n");
}


//******************************************************************************************
// the forward system is split into 3 subsystems: set_forward_plan, _forward and remove_pkts
// set_forward_plan on *ingress* port
template <typename T>
void printBits(T value) {
    for(CountT i = sizeof(T) * 8 - 1; i >= 0; i--) {
        if(value & (1 << i)) printf("1");
        else printf("0");
    }
    printf("\n");
}

inline void set_forward_plan(Engine &ctx,
                            PortID &_port_id,
                            SwitchID &_switch_id,
                            PktBuf &_queue,
                            ForwardPlan &_forward_plan,
                            SimTime & _sim_time,
                            SimTimePerUpdate &_sim_time_per_update)
{   
    if (_port_id.port_id == 0) {
        printf("*******Enter into set_forward_plan*********\n");
    }
    // int32_t _numEPort = ctx.data().numEPort;
    // fib_table lookup
    // FIXME 
    // if (_port_id.port_id == 0) {
    //     for(CountT i = 0; i < SS_LINK_DELAY/PACKET_PROCESS_TIME; i++) {
    //         _forward_plan.forward_plan[i % _numEPort] = _forward_plan.forward_plan[i % _numEPort] | (1<<i); 
    //     }
    // }
    // else {
    //      for(CountT i = 0; i < SS_LINK_DELAY/PACKET_PROCESS_TIME; i++) {
    //         if(i < 3) {
    //             _forward_plan.forward_plan[0] = _forward_plan.forward_plan[0] | (1<<i);
    //         }
    //         else {
    //             _forward_plan.forward_plan[1] = _forward_plan.forward_plan[1] | (1<<i);
    //         }
    //     }
    // }

    // for (int32_t i = 0; i < SS_LINK_DELAY/PACKET_PROCESS_TIME; i++) {
    //     Pkt packet = {0, 0, 0, 0, 0, 0};
    //     fetch_pkt(_queue, i, packet);
    //     int32_t sw_idx = _switch_id.switch_id;
    //     //printf("sw_idx: %d, ", sw_idx);
    //     Entity sw_id = ctx.data()._switch[sw_idx];
    //     int32_t next_hop_link_idx = ctx.get<FIBTable>(sw_id).fib_table[packet.dest_id][packet.flow_id];
    //     //int32_t next_hop_link_idx = ctx.get<FIBTable>(sw_id).fib_table[0][0];
    //     //int32_t next_hop_link_idx = 0;
    //     _forward_plan.forward_plan[next_hop_link_idx] = _forward_plan.forward_plan[next_hop_link_idx] | (1<<i); 
    // }

    //reset
    for (uint8_t i = 0; i < INPORT_NUM; i++) {_forward_plan.forward_plan[i] = 0;}

    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    
    if (get_queue_len(_queue) <= 0) return;
    
    int16_t idx = 0;
    while(true) {
        Pkt packet = {0, 0, 0, 0, 0, 0};
        // fetch packet from buffer while checking whether buffer is empty
        if(!fetch_pkt(_queue, idx, packet)) break;

        if(packet.enqueue_time >= end_time) break;

        printf("packet.enqueue_time: %lld, end_time: %lld\n", packet.enqueue_time, end_time);
        int32_t sw_idx = _switch_id.switch_id;
        //printf("sw_idx: %d, ", sw_idx);
        Entity sw_id = ctx.data()._switch[sw_idx];
        int32_t next_hop_link_idx = ctx.get<FIBTable>(sw_id).fib_table[packet.dest_id][packet.flow_id];
        _forward_plan.forward_plan[next_hop_link_idx] = _forward_plan.forward_plan[next_hop_link_idx] | (1<<idx);

        idx += 1;
    }
    //_sim_time.sim_time = end_time; // FIXME
    //if (_port_id.port_id != 0 || _switch_id.switch_id != 0) return;

    printf("forward plan of the %d-th ingress port: \n", _port_id.port_id);
    for(CountT i = 0; i < INPORT_NUM/4; i++) {
        //printf("towards %d-th egress port: %d\n", i, _forward_plan.forward_plan[i]);
        printf("towards %d-th egress port:", i);
        printBits<uint16_t>(_forward_plan.forward_plan[i]);
    }
    printf("\n");
}


// on *egress* port
// each e_port fetches packets from in_ports
inline void _forward(Engine &ctx,
                    SchedTrajType sched_traj_type,
                    PortID &_port_id,
                    PktBuf &_queue,
                    SwitchID &_switch_id,
                    SimTime &_sim_time,
                    SimTimePerUpdate &_sim_time_per_update)
{
    if (_port_id.port_id == 0) {
        printf("*******Enter into _forward sys*********\n");
    }
    Entity in_port;
    uint16_t fwd_bitmap;
    int32_t _numInPort = ctx.data().numInPort;

    // FIXME: modify it to support the any shape of switch
    // for(CountT i = _switch_id.switch_id*K_ARY; i < (_switch_id.switch_id+1)*K_ARY; i++) {
    //     in_port = ctx.data().inPorts[i];
    //     // if (ctx.get<SwitchID>(in_port).switch_id != _switch_id.switch_id) continue;
    //     fwd_bitmap = ctx.get<ForwardPlan>(in_port).forward_plan[_port_id.port_id];
    //     // printf("forward_plan of the %d-th inport toward the %d-th eport:: \n", i, _port_id.port_id);
    //     // printBits(fwd_bitmap);
    //     PktBuf pkt_buf = ctx.get<PktBuf>(in_port);
    //     // printf("-inport(_forward), queue_head: %d, queue_tail: %d\n", pkt_buf.head, pkt_buf.tail);
    //     for (CountT num = 0; num < SS_LINK_DELAY/PACKET_PROCESS_TIME; num++) {
    //         if ((fwd_bitmap & (1<<num)) != 0) { // the packet should be forward to the e_port
    //             // printf("the %d-th packet needs to be forward\n", num);
    //             Pkt packet = {0, 0, 0, 0, 0, 0};
    //             //assert(_dequeue(pkt_buf, packet != 0));

    //             fetch_pkt(pkt_buf, num, packet);
    //             // printf("%d", packet.flow_id); 
    //             _enqueue(_queue, packet);
    //         }
    //     }
    // }

    // FIXME: modify it to support the any shape of network topo
    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    
    for(CountT i = _switch_id.switch_id*K_ARY; i < (_switch_id.switch_id+1)*K_ARY; i++) {

        in_port = ctx.data().inPorts[i];
        fwd_bitmap = ctx.get<ForwardPlan>(in_port).forward_plan[_port_id.port_id];
        PktBuf pkt_buf = ctx.get<PktBuf>(in_port);

        //if (get_queue_len(_queue) <= 0) return;
        int16_t num = 0;
        int16_t count_bit = sizeof(fwd_bitmap)*8;
        while(count_bit--) {
            if ((fwd_bitmap & (1<<num)) != 0) { // the packet should be forward to the e_port
                // printf("the %d-th packet needs to be forward\n", num);
                Pkt packet = {0, 0, 0, 0, 0, 0};
                //assert(_dequeue(pkt_buf, packet) != 0);
                fetch_pkt(pkt_buf, num, packet);
                // printf("%d", packet.flow_id); 
                _enqueue(_queue, packet);
                num += 1;
            }

        }
    }
    //???????
    //_sim_time.sim_time = end_time; // FIXME
}


// on *ingress* port
//remove batch packets from ingress port buffer
inline void remove_pkts(Engine &ctx,
                        PortID &_port_id,
                        PktBuf &_queue,
                        ForwardPlan &_forward_plan,
                        SimTime &_sim_time,
                        SimTimePerUpdate &_sim_time_per_update)
{   
    // Pkt packet;
    // for (CountT num = 0; num < SS_LINK_DELAY/PACKET_PROCESS_TIME; num++) {
    //     if ((_forward_plan.forward_plan[_port_id.port_id] & (1<<num)) != 0) {
    //         _dequeue(_queue, packet);
    //     }
    // }
    if (_port_id.port_id == 0) {
        printf("*******Enter into remove_pkts sys*********\n");
    }
    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    
    if (get_queue_len(_queue) <= 0) return;
    
    int16_t idx = 0;
    Pkt packet = {0, 0, 0, 0, 0, 0};
    while(true) {
        // fetch packet from buffer while checking whether buffer is empty
        if(!fetch_pkt(_queue, idx, packet)) break;
        if(packet.enqueue_time >= end_time) break;
        _dequeue(_queue, packet);
    }
    //???????
    _sim_time.sim_time = end_time; // FIXME
}
//========================================================================================================


void transmit(Engine &ctx,
              SchedTrajType sched_traj_type,
              PortType port_type,
              PortID &_port_id,
              NextHop &_next_hop,
              NextHopType next_hop_type,
              PktBuf &_queue,
              TXHistory &_tx_history,
              SSLinkDelay &_ss_link_delay,
              LinkRate &_link_rate,
              SimTime &_sim_time,
              SimTimePerUpdate &_sim_time_per_update)
{
    // Implementation of the Insertion Sort algorithm to sort packets in terms of enqueue_time
    // auto insertionSort = [](Pkt *data, int32_t length) {
    //     int32_t i, j;
    //     Pkt key;
    //     for (i = 1; i < length; i++) {
    //         key = data[i];
    //         j = i - 1;

    //         // Move elements of data[0..i-1] that are greater than key.enqueue_time to one position ahead of their current position
    //         while (j >= 0 && data[j].enqueue_time > key.enqueue_time) {
    //             data[j + 1] = data[j];
    //             j = j - 1;
    //         }
    //         data[j + 1] = key;
    //     }
    // };
    auto insertionSort = [](PktBuf data) {
        int32_t i, j;
        Pkt key;
        int16_t len = get_queue_len(data);
        for (i = 1; i < len; i++) {
            key = get_pkt(data, i);
            j = i - 1;

            // Move elements of data[0..i-1] that are greater than key.enqueue_time to one position ahead of their current position
            while (j >= 0 && get_pkt(data, j).enqueue_time > key.enqueue_time) {
                get_pkt(data, j+1) = get_pkt(data, j);

                j = j - 1;
            }
            get_pkt(data, j+1) = key;
        }
    };

    if (_port_id.port_id == 0) {
        printf("*******Enter into transmit sys*********\n");
    }
    if (get_queue_len(_queue) <= 0) return;
    // Sorting the packets in the queue using Insertion Sort
    //insertionSort(_queue.pkts + _queue.head, _queue.tail - _queue.head);
    insertionSort(_queue);

    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    
    int16_t idx = 0;
    Pkt packet;
    while(true) {
        if (_sim_time.sim_time >= end_time) break;
        // fetch packet from buffer, while checking whether buffer is empty
        Pkt packet = {0, 0, 0, 0, 0, 0};
        //if(!fetch_pkt(_queue, idx, packet)) break; 
        if(!_dequeue(_queue, packet)) break; // dequeue a packet and remove it
        if(packet.enqueue_time >= end_time) break;

        int16_t inst_queue_len = 0; // record the queue length when a packet comes in real time
        int16_t tx_his_idx = 0;
        for (int i = get_queue_len(_tx_history); i > 0; i--) {
            TXElem tx_elem = {0, 0};
            fetch_pkt(_tx_history, tx_his_idx, tx_elem);
            if (packet.enqueue_time < tx_elem.dequeue_time) {
                inst_queue_len += tx_elem.pkt_size;
            }
            else {
                break;
            }
            tx_his_idx += 1;
        }

        clear_queue(_tx_history);

        packet.dequeue_time = _sim_time.sim_time + (packet.pkt_size*8*1.0/_link_rate.link_rate)*(1000*1000*1000);
        packet.enqueue_time = packet.dequeue_time + _ss_link_delay.SS_link_delay;

        _sim_time.sim_time = packet.dequeue_time;
        
        //move packet from this egress port to next hop port / the des host
        Entity next_hop_ett;
        if (next_hop_type == NextHopType::SWITCH) {
            next_hop_ett = ctx.data().inPorts[_next_hop.next_hop];
        }
        else {
            next_hop_ett = ctx.data().receiver[_next_hop.next_hop];
        }
        _enqueue(ctx.get<PktBuf>(next_hop_ett), packet);
    }
    //???????
    _sim_time.sim_time = end_time; // FIXME


    // printf("after sorted, pkt_buf: ");
    // for(CountT idx = _queue.head; idx < _queue.tail; idx++) {
    //     printf("%d ", _queue.pkts[idx].enqueue_time);
    // }
    // printf("\n");
}



// // create new packets and move them into direct-connected ingress ports
// inline void receive(Engine &ctx, ReceiverID &_receiver_id, IngressPortID &_ingress_port_id) {
//     Entity e_port = ctx.data().ePorts[_ingress_port_id.ingress_port_id];
//     // udp sending 
//     printf("*******Enter into receive*********\n");
//     // int32_t pkt_num = SS_LINK_DELAY/PACKET_PROCESS_TIME;
//     // for (int32_t idx = 0; idx < pkt_num; idx++) {
//     //     Pkt packet = {0, 0, 0, 0, 0};
//     //     _dequeue(ctx.get<PktBuf>(e_port), packet) != 0;
//     // }
    
//     //printf("pkt_buf.tail, %d; pkt_buf.head, %d in eport\n", ctx.get<PktBuf>(e_port).tail, ctx.get<PktBuf>(e_port).head);

// }


inline void receive(Engine &ctx, ReceiverID &_receiver_id, NextHop &_next_hop, PktBuf &_queue,\
                    NICRate &_nic_rate, HSLinkDelay &_HS_link_delay, SimTime &_sim_time, \
                    SimTimePerUpdate &_sim_time_per_update) {
    // udp receive
    if (_receiver_id.receiver_id == 0) {
        printf("*******Enter into receive*********\n");
    }
    Entity in_port = ctx.data().inPorts[_next_hop.next_hop];
    //printf("send new packets into the buffer of directed-connected in_port\n");
    int64_t end_time = _sim_time.sim_time + _HS_link_delay.HS_link_delay;
    int16_t ack_pkt_size = 64; //bytes
    // int16_t flow_id = 0;
    // int16_t count_pkt = 0;
    Pkt packet;
    while (_sim_time.sim_time < end_time) {
        if(!_dequeue(_queue, packet)) break; // dequeue a packet and remove it
        if(packet.enqueue_time >= end_time) break;

        // //printf("_sim_time.sim_time: %lld, end_time: %lld\n", _sim_time.sim_time, end_time);
        // int64_t dequeue_time = _sim_time.sim_time + (pkt_size*8*1.0/_nic_rate.nic_rate)*(1000*1000*1000); // unit is ns
        // //printf("dequeue_time: %lld\n", dequeue_time);
        // int64_t enqueue_time = dequeue_time + _HS_link_delay.HS_link_delay;

        // Pkt packet = {_sender_id.sender_id, receiver_id, ack_pkt_size, flow_id, enqueue_time, dequeue_time};
        // _enqueue(ctx.get<PktBuf>(in_port), packet);
        // count_pkt += 1;
        // _sim_time.sim_time = dequeue_time;
    }
    _sim_time.sim_time = end_time;
}





inline void _debug_e_port(Engine &ctx,
                        SchedTrajType sched_traj_type,
                        PortType port_type,
                        PortID &_port_id,
                        PktBuf &_queue)
{
    // printf("_debug_e_port  of %d:\n", _port_id.port_id);
    // printf("queue_head: %d, queue_tail: %d\n", _queue.head, _queue.tail);
    // if (_port_id.port_id == 1) {
    //     return;
    // }
    // // while(1) {
    // //     Pkt packet;
    // //     if (_dequeue(_queue, packet) != false)
    // //         printf("%d ", packet.enqueue_time);
    // //     else
    // //         break;
    // // }

    // for(CountT idx = _queue.head; idx < _queue.tail; idx++) {
    //     printf("%d ", _queue.pkts[idx].enqueue_time);
    // }
    // printf("\n");
}


void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
{
    printf("*******Enter into setupTasks*********\n");
    auto send_sys = builder.addToGraph<ParallelForNode<Engine, send, SenderID, NextHop, NICRate, HSLinkDelay, SimTime, SimTimePerUpdate>>({});

    auto set_forward_plan_sys = builder.addToGraph<ParallelForNode<Engine, set_forward_plan, PortID, \
                                                   SwitchID, PktBuf, ForwardPlan, SimTime, SimTimePerUpdate>>({send_sys});

    auto forward_sys = builder.addToGraph<ParallelForNode<Engine, _forward, SchedTrajType, \
                                          PortID, PktBuf, SwitchID, SimTime, SimTimePerUpdate>>({set_forward_plan_sys});

    auto remove_pkts_sys = builder.addToGraph<ParallelForNode<Engine, remove_pkts, \
                                              PortID, PktBuf, ForwardPlan, SimTime, SimTimePerUpdate>>({forward_sys}); 
                   
    // auto debug_sys = builder.addToGraph<ParallelForNode<Engine, _debug_e_port, \
    //                                     SchedTrajType, PortType, PortID, PktBuf, TXHistory, 
    //                                     SSLinkDelay, LinkRate, SimTime>>({remove_pkts_sys}); 

    auto transmit_sys = builder.addToGraph<ParallelForNode<Engine, transmit, SchedTrajType, \
                                           PortType, PortID, NextHop, NextHopType, PktBuf, TXHistory, \
                                           SSLinkDelay, LinkRate, SimTime, SimTimePerUpdate>>({remove_pkts_sys}); 

    auto receive_sys = builder.addToGraph<ParallelForNode<Engine, receive, ReceiverID, NextHop, PktBuf, \
                                          NICRate, HSLinkDelay, SimTime, SimTimePerUpdate>>({transmit_sys}); 

}


Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    Entity agent = ctx.makeEntity<Agent>();
    ctx.get<Action>(agent) = Action::None;
    ctx.get<GridPos>(agent) = GridPos {
        grid->startY,
        grid->startX,
    };
    ctx.get<Reward>(agent).r = 0.f;
    ctx.get<Done>(agent).episodeDone = 0.f;
    ctx.get<CurStep>(agent).step = 0;

    //***************************************************
    // for the router forwarding function, of GPU_acclerated DES
    numInPort = 0;
    numEPort = 0;

    numSender = 0;
    numReceiver = 0;
    numSwitch = 0;  

    generate_switch(ctx, K_ARY);
    generate_in_port(ctx, K_ARY);
    generate_e_port(ctx, K_ARY);
    generate_host(ctx, K_ARY);

    printf("Gen configuration is as following: \n");
    printf("Total %d switches, %d in_ports, %d e_ports, %d hosts", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, ctx.data().numSender);

    //generate_fib_table(ctx, K_ARY);
    //***************************************************

}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
