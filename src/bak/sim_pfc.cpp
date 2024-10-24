#include "sim.hpp"
// #include "dcqcn.hpp"
#include "build_topo.hpp"
// #include "myQueue.hpp"

#include <madrona/mw_gpu_entry.hpp> 
#include <atomic>

// for conditional compilination
#include <type_traits>
#include <utility>

using namespace madrona;
using namespace madrona::math;

#define PRINT_PKT_LOG 1

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
    
    //pfc
    registry.registerComponent<PFCState>();
    registry.registerComponent<PktQueue>(); //PktQueue is composed of multiple PktBuf and a PFCState array
     
    registry.registerComponent<SimTime>();
    registry.registerComponent<SimTimePerUpdate>();

    registry.registerComponent<SendFlowQueue>();
    registry.registerComponent<SenderID>();
    registry.registerComponent<EgressPortID>();

    registry.registerArchetype<Sender>();

    registry.registerComponent<RecvFlowQueue>();
    registry.registerComponent<ReceiverID>();
    registry.registerComponent<IngressPortID>();
    
    registry.registerArchetype<Receiver>();

    registry.registerComponent<SwitchID>();
    registry.registerComponent<SwitchType>();
    registry.registerComponent<FIBTable>();
    registry.registerComponent<QueueNumPerPort>();
    
    registry.registerArchetype<Switch>();

    registry.registerComponent<LinkRate>();
    registry.registerComponent<SSLinkDelay>();

    registry.registerComponent<SchedTrajType>();
    registry.registerComponent<PortType>();
    registry.registerComponent<LocalPortID>();
    registry.registerComponent<GlobalPortID>();

    registry.registerComponent<TXHistory>();  

    registry.registerComponent<QLen>();
    
    registry.registerComponent<ForwardPlan>();

    registry.registerComponent<Seed>();

    registry.registerArchetype<IngressPort>();
    registry.registerArchetype<EgressPort>();

    // registry.registerComponent<FlowID>();    
    // registry.registerArchetype<Flow_Queue>();
}


//random generator (Linear Congruential Generator)
class LCG {
private:
    unsigned long a;
    unsigned long c;
    unsigned long modulus;
    unsigned long seed;

public:
    LCG(unsigned long _a, unsigned long _c, unsigned long _modulus, unsigned long _seed)
        : a(_a), c(_c), modulus(_modulus), seed(_seed) {}

    unsigned long next() {
        seed = (a * seed + c) % modulus;
        return seed;
    }

    void set_seed(unsigned long _seed) {
        seed = _seed;
    }
};

//******************************************
//the operation set for queue managing


// Helper type trait to check if T has a member named cur_bytes
template <typename T, typename = void>
struct has_cur_bytes : std::false_type {};

template <typename T>
struct has_cur_bytes<T, std::void_t<decltype(std::declval<T>().cur_bytes)>> : std::true_type {};


template<typename T, size_t N>
size_t getArrayLength(T (&)[N]) {
    return N;
}

template<typename T>
inline Pkt& get_elem(T& queue, int32_t idx) 
{
    return queue.pkts[(queue.head+idx) % getArrayLength(queue.pkts)];
}


template<typename T1, typename T2>
inline bool _dequeue(T1& queue, T2& pkt) 
{
    if (queue.cur_num == 0) {
        // The queue is empty, cannot dequeue
        return false;
    }

    pkt = queue.pkts[queue.head];
    //queue.head = (queue.head + 1) % (getArrayLength(queue.pkts)); // use modulo operation to wrap around
    queue.head = (queue.head + 1) % PKT_BUF_LEN;
    queue.cur_num -= 1;

    if constexpr (has_cur_bytes<T1>::value) {
        queue.cur_bytes -= pkt.ip_pkt_len;
    }
    // queue.cur_bytes -= pkt.ip_pkt_len;

    return true; 
}

template<typename T1, typename T2>
inline bool fetch_elem(T1& queue, int32_t idx, T2& elem) 
{
    if (queue.cur_num == 0) 
        return false; // The queue is empty, cannot dequeue
    //elem = queue.pkts[(queue.head+idx) % getArrayLength(queue.pkts)]; // use modulo operation to wrap around
    elem = queue.pkts[(queue.head+idx) % PKT_BUF_LEN];
    //queue.head = (queue.head + 1) % (PKT_BUF_LEN); // use modulo operation to wrap around

    return true; 
}

template<typename T1, typename T2>
inline void _enqueue(T1 &queue, T2 pkt) {
    //queue.pkts[queue.tail] = {pkt.src, pkt.dst, pkt.pkt_size, pkt.flow_id, pkt.enqueue_time, pkt.dequeue_time};
    queue.pkts[queue.tail] = pkt;
    //queue.tail = (queue.tail + 1) % getArrayLength(queue.pkts);
    queue.tail = (queue.tail + 1) % PKT_BUF_LEN;
    queue.cur_num += 1;

    if constexpr (has_cur_bytes<T1>::value) {
        queue.cur_bytes += pkt.ip_pkt_len;
    }
    // queue.cur_bytes += pkt.ip_pkt_len;
}

template<typename T>
inline uint16_t get_queue_len(T &queue) {
    return queue.cur_num;
}

template<typename T>
inline void clear_queue(T &queue) {
    queue.head = 0;
    queue.tail = 0;
    queue.cur_num = 0;

    if constexpr (has_cur_bytes<T>::value) {
        queue.cur_bytes = 0;
    }
    // queue.cur_bytes = 0;
}

// remove the range of queue between start and end
template<typename T1, typename T2>
inline void rm_queue_range(T1 &queue, T2 pkt, uint16_t start, uint16_t end) {
    for (uint16_t i = start; i < end; i++) {
        _dequeue(queue, pkt);
    }
}


///////////////////////////////////////////////////////////////////////////////////
// DCQCN CC logic
void dcqcn_rate_increase(SendFlow &flow, long curSimTime)
{
    if (flow.dcqcn_IncreaseStageCount < flow.dcqcn_RecoveryStageThreshold)
    {
        // fast recovery stage
        // update current rate
        flow.m_rate = flow.m_rate / 2 + flow.tar_rate / 2;

        printf("In fast recovery stage. sender: %d, sim_time: %ld, target rate is %.3lf Gbps, and current rate is %.3lf Gbps\n", flow.src, curSimTime, flow.tar_rate/(1000*1000*1000.0), flow.m_rate/(1000*1000*1000.0));
    }
    else if (flow.dcqcn_IncreaseStageCount == flow.dcqcn_RecoveryStageThreshold)
    {
        // additive increase stage
        // target rate increases with fixed AI sending rate
        flow.tar_rate += flow.dcqcn_AIRate;

        // if target rate is bigger than the device rate, set to device rate
        if (flow.tar_rate > NIC_RATE) {
            flow.tar_rate = NIC_RATE;
        }
        // update current rate
        flow.m_rate = flow.m_rate / 2 + flow.tar_rate / 2;

        printf("In additive increase stage. sender: %d, sim_time: %ld, target rate is %.3lf Gbps, and current rate is %.3lf Gbps\n", flow.src, curSimTime, flow.tar_rate/(1000*1000*1000.0), flow.m_rate/(1000*1000*1000.0));
    }
    else
    {
        // hyper increase stage
        // target rate increases with fixed AI sending rate
        flow.tar_rate += flow.dcqcn_HAIRate;

        // if target rate is bigger than the device rate, set to device rate
        if (flow.tar_rate > NIC_RATE) {
            flow.tar_rate = NIC_RATE;
        }
        // update current rate
        flow.m_rate = flow.m_rate / 2 + flow.tar_rate / 2;

        printf("In hyper increase stage. sender: %d, sim_time: %ld, target rate is %.3lf Gbps, and current rate is %.3lf Gbps\n", flow.src, curSimTime, flow.tar_rate/(1000*1000*1000.0), flow.m_rate/(1000*1000*1000.0));
    }

    // increase the stage count
    flow.dcqcn_IncreaseStageCount++;
}

void dcqcn_alpha_update(SendFlow &flow)
{
    // if we receive CNP packet, increase value of alpha
    if (flow.CNPState == true)
    {
        flow.dcqcn_Alpha = (1 - flow.dcqcn_G) * flow.dcqcn_Alpha + flow.dcqcn_G;
        // flow.CNPState = false;
    }
    // else decrease the value of alpha
    else
    {
        flow.dcqcn_Alpha = (1 - flow.dcqcn_G) * flow.dcqcn_Alpha;
    }
}

// decrease stage of sending rate in DCQCN, when received ECE packet
void dcqcn_rate_decrease(SendFlow &flow, long curSimTime) {
    // if (flow.ECNState != ECNState.ECE_RECV)
    if (flow.CNPState != true)
        return;
    // record current rate to later fast recovery
    // printf(" original rate: %.3lf, dcqcn_Alpha: %.3lf\n", flow.m_rate/(1000*1000*1000.0), flow.dcqcn_Alpha);
    
    flow.tar_rate = flow.m_rate;
    flow.m_rate = (long int)(flow.m_rate * (1.0 - flow.dcqcn_Alpha / 2.0));
    flow.dcqcn_IncreaseStageCount = 0;

    // printf("DCQCN decrease rate to %.3lf Gbps\n", flow.m_rate/(1000*1000*1000.0));

    printf("In decrease stage. sender: %d, sim_time: %ld, target rate is %.3lf Gbps, and current rate is %.3lf Gbps\n", flow.src, curSimTime, flow.tar_rate/(1000*1000*1000.0), flow.m_rate/(1000*1000*1000.0));
}

// reset the alpha update and rate increase timer, when rate is decreased
void dcqcn_timer_reset_if_decrease(SendFlow &flow, long curSimTime) {

    // if rate isn't decreased
    if (!flow.dcqcn_RateIsDecreased)
        return;

    // since we decrease the rate, we need to reset rate increase timer
    flow.dcqcn_AlphaUpdateNextTime = flow.dcqcn_RateIncreaseNextTime = curSimTime + flow.dcqcn_RateIncreaseInterval;
    //
    flow.dcqcn_RateIsDecreased = false;
}

// 
void dcqcn_timer_rate_increase(SendFlow &flow, long curSimTime) {
    // if current simulation time is smaller than the next rate increase time,
    // should not tigger this function
    if (curSimTime <= flow.dcqcn_RateIncreaseNextTime)
        return;

    // schedule the next rate increase timer
    flow.dcqcn_RateIncreaseNextTime += flow.dcqcn_RateIncreaseInterval;

    // increase sending rate
    dcqcn_rate_increase(flow, curSimTime);

}

void dcqcn_timer_alpha_update(SendFlow &flow, long curSimTime) {

    // if current simulation time is smaller than the next rate decrease time,
    // should not tigger this function
    if (curSimTime <= flow.dcqcn_AlphaUpdateNextTime)
        return;
    // schedule the next alpha update timer
    flow.dcqcn_AlphaUpdateNextTime += flow.dcqcn_AlphaUpdateInterval;

    dcqcn_alpha_update(flow);
}
///////////////////////////////////////////////////////////////////////////////////


inline void PrintPkt(Pkt p, char *desc) {
    // if (p.sq_num != 1664400) return; //1664400
    if (p.pkt_type == PktType::DATA)
        printf("pkt type: DATA\n");
    else if (p.pkt_type == PktType::ACK)
        printf("pkt type: ACK\n");
    else if (p.pkt_type == PktType::PFC_PAUSE)
        printf("pkt type: PFC_PAUSE\n");
    else if (p.pkt_type == PktType::PFC_RESUME)
        printf("pkt type: PFC_RESUME\n");
    else if (p.pkt_type == PktType::NACK)
        printf("pkt type: NACK\n");
    else
        printf("pkt type: CNP\n");
    printf("%s:: src: %d, dst: %d, ecn: %d, payload_len: %d, sq_num: %ld, enqueue_time: %ld, dequeue_time: %ld\n", desc, p.src, p.dst, p.ecn, p.payload_len, p.sq_num, p.enqueue_time, p.dequeue_time);
}

template<typename T>
inline void PrintQueue(T &queue) {
    printf("packet buffer is: \n");
    for (uint16_t i = 0; i < queue.cur_num; i++) {
        Pkt pkt = get_elem(queue, i);
        PrintPkt(pkt, "");
    }
    printf("\n");
}

inline void create_pkt(Pkt &pkt,
    uint32_t src,   // PFC state for the PFC frame 
    uint32_t dst,  // PFC_FOR_UPSTREAM for the PFC frame 
    uint16_t l4_port,
    uint16_t header_len,
    uint16_t payload_len, 
    uint16_t ip_pkt_len,
    int64_t sq_num,
    uint16_t flow_id,  // also the idx of priority queue for the PFC frame 
    int64_t enqueue_time,
    int64_t dequeue_time,
    uint8_t flow_priority,
    PktType pkt_type, 
    uint32_t _path[MAX_PATH_LEN],
    uint8_t path_len,
    ECN_MARK ecn)
{
    pkt.src = src;
    pkt.dst = dst;
    pkt.l4_port = l4_port;
    pkt.header_len = header_len;
    pkt.payload_len = payload_len;
    pkt.ip_pkt_len = ip_pkt_len;
    pkt.sq_num = sq_num;
    pkt.flow_id = flow_id;
    pkt.enqueue_time = enqueue_time;
    pkt.dequeue_time = dequeue_time;
    pkt.flow_priority = flow_priority;
    pkt.pkt_type = pkt_type;
    memcpy(pkt.path, _path, MAX_PATH_LEN * sizeof(uint32_t));
    pkt.path_len = path_len;
    pkt.ecn = ecn;
}

// inline void create_entt(Engine &ctx, FlowID &_flow_id) {
//     Entity new_flow = ctx.makeEntity<Flow>();
//     ctx.get<FlowID>(new_flow) = ;
// }


inline void send(Engine &ctx, 
                 SenderID &_sender_id, NextHop &_next_hop,
                 NICRate &_nic_rate, HSLinkDelay &_HS_link_delay, 
                 SimTime &_sim_time, SimTimePerUpdate &_sim_time_per_update, 
                 SendFlowQueue &_send_flow_queue, PktBuf &_recv_ack_buf) {

    // if (_sender_id.sender_id != 0 && _sender_id.sender_id != 2) {
    // if (_sender_id.sender_id != 2) {
    //     return;
    // }
    Entity in_port = ctx.data().inPorts[_next_hop.next_hop];
    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    if (_sender_id.sender_id == 0 || _sender_id.sender_id == 1) {
        printf("\n*******Enter into send*********\n");
        printf("start _sim_time.sim_time: %ld\n", _sim_time.sim_time);
        printf("end_time: %ld\n", end_time);
        // printf("cc target rate: %.3lf, cc cur rate: %.3lf\n", _send_flow_queue.send_flow[0].tar_rate/(1000*1000*1000.0), _send_flow_queue.send_flow[0].m_rate/(1000*1000*1000.0));
    }

    uint16_t flow_id = 0;
    uint16_t header_len = 40; 
    uint8_t flow_priority = 0;
    uint32_t path[MAX_PATH_LEN];
    memset(path, 0, MAX_PATH_LEN*sizeof(MAX_PATH_LEN));

    int16_t sent_bytes = 0; 
    while (_sim_time.sim_time < end_time) {
        //printf("send sim_time: %lld\n", _sim_time.sim_time);   
        //printf("_sim_time.sim_time < end_time");
        Pkt ack_p;
        // if (_sender_id.sender_id == 0) { 
        //     printf("sim_time: %d, nxt_pkt_event: %d\n", _sim_time.sim_time, _send_flow_queue.send_flow[0].nxt_pkt_event);
        // }
        // new arrival of ack packet
        uint16_t ack_pkt_cnt = 0;
        while (fetch_elem(_recv_ack_buf, 0, ack_p)) {  // if ack pkt is available
            if ((ack_p.enqueue_time >= end_time) && (ack_p.enqueue_time > _send_flow_queue.send_flow[0].nxt_pkt_event && _send_flow_queue.send_flow[0].snd_nxt != _send_flow_queue.send_flow[0].flow_size)) {
                break;
            }
            // if (_send_flow_queue.send_flow[0].snd_nxt >= _send_flow_queue.send_flow[0].flow_size) {
            //     break;
            // }
            // if (_sender_id.sender_id == 0) { 
            //     //printf("ack_p.enqueue_time: %ld, nxt_pkt_event: %ld\n", ack_p.enqueue_time, _send_flow_queue.send_flow[0].nxt_pkt_event);
            // }

            _dequeue(_recv_ack_buf, ack_p);

            if (ack_p.dst == _sender_id.sender_id) {// reaction for ACK/NACK packets
                #if PRINT_PKT_LOG
                PrintPkt(ack_p, "*ACK received by sender: ");
                #endif
                if (ack_p.pkt_type == PktType::ACK) { // ACK
                    _send_flow_queue.send_flow[0].snd_una = ack_p.sq_num;
                    printf("sender id: %d, variable set: snd_una: %ld\n", _send_flow_queue.send_flow[0].src, _send_flow_queue.send_flow[0].snd_una);

                    if (_send_flow_queue.send_flow[0].snd_una >= _send_flow_queue.send_flow[0].flow_size) {
                        _send_flow_queue.send_flow[0].flow_state = FlowState::COMPLETE;
                        // _send_flow_queue.send_flow[0].nxt_pkt_event = INT64_MAX;
                        if (_send_flow_queue.send_flow[0].flow_size != 0) {
                            printf("sender id: %d, flow is finished\n", _send_flow_queue.send_flow[0].src, _send_flow_queue.send_flow[0].snd_una);
                        }
                        break;
                    }

                    _send_flow_queue.send_flow[0].last_ack_timestamp = _sim_time.sim_time;
                }
                else if (ack_p.pkt_type == PktType::NACK) { // NACK
                    _send_flow_queue.send_flow[0].snd_nxt = ack_p.sq_num;
                    _send_flow_queue.send_flow[0].snd_una = ack_p.sq_num;
                    //printf("variable set: snd_una: %ld, snd_nxt: %ld\n", _send_flow_queue.send_flow[0].snd_una, _send_flow_queue.send_flow[0].snd_nxt);

                    _send_flow_queue.send_flow[0].last_ack_timestamp = _sim_time.sim_time;
                }
                else if (ack_p.pkt_type == PktType::CNP){ // CNP
                    _send_flow_queue.send_flow[0].CNPState = true;
                    dcqcn_rate_decrease(_send_flow_queue.send_flow[0], _sim_time.sim_time);
                    dcqcn_alpha_update(_send_flow_queue.send_flow[0]);
                    _send_flow_queue.send_flow[0].CNPState = false;
                    //record the rate is decreased
                    //use it to determine whether up
                    _send_flow_queue.send_flow[0].dcqcn_RateIsDecreased = true;
                }
                else { //PFC pause/resume packet 
                    if (ack_p.pkt_type == PktType::PFC_PAUSE) {
                        _send_flow_queue.send_flow[0].pfc_state = PFCState::PAUSE;
                    }
                    else {
                         _send_flow_queue.send_flow[0].pfc_state = PFCState::RESUME;
                    }
                }
            }
            else { // the ACK/NACK packet is not for me, send it out to the receiver
                _enqueue(ctx.get<PktBuf>(in_port), ack_p);
            }
            _sim_time.sim_time += 1; // increase 1 ns
            ack_pkt_cnt += 1;
        }

        // DCQCN timer function
        // the order should be: check if decrease, update alpha, increase send rate
        dcqcn_timer_reset_if_decrease(_send_flow_queue.send_flow[0], _sim_time.sim_time);
        dcqcn_timer_alpha_update(_send_flow_queue.send_flow[0], _sim_time.sim_time);
        dcqcn_timer_rate_increase(_send_flow_queue.send_flow[0], _sim_time.sim_time);


        // if (_sender_id.sender_id == 0) {
        //     printf("snd_una: %ld, snd_nxt: %ld\n", _send_flow_queue.send_flow[0].snd_una, _send_flow_queue.send_flow[0].snd_nxt);
        // }

        if (_send_flow_queue.send_flow[0].snd_una >= _send_flow_queue.send_flow[0].flow_size) {
            _send_flow_queue.send_flow[0].flow_state = FlowState::COMPLETE;
            break;
        }

        if (_sim_time.sim_time > _send_flow_queue.send_flow[0].nxt_pkt_event) {
            _send_flow_queue.send_flow[0].nxt_pkt_event = _sim_time.sim_time;
        }
        else {
            _sim_time.sim_time = _send_flow_queue.send_flow[0].nxt_pkt_event;
        }
        // } // cannot handle when _sim_time.sim_time == nxt_pkt_event

        if ((_sim_time.sim_time - _send_flow_queue.send_flow[0].last_ack_timestamp) >= RETRANSMIT_TIMER) {
            _send_flow_queue.send_flow[0].snd_nxt = _send_flow_queue.send_flow[0].snd_una;
            _send_flow_queue.send_flow[0].last_ack_timestamp = _sim_time.sim_time;
        }

        // there must no new send event in this time frame
        if (ack_pkt_cnt == 0 && _send_flow_queue.send_flow[0].snd_nxt == _send_flow_queue.send_flow[0].flow_size) {break;}

        if (_send_flow_queue.send_flow[0].pfc_state == PFCState::PAUSE) {
            printf("In send phase: PFCState::PAUSE\n");
        }

        if (_send_flow_queue.send_flow[0].flow_state == FlowState::UNCOMPLETE && _send_flow_queue.send_flow[0].pfc_state == PFCState::RESUME) { // the flow is unfinished
            // flow is not finished
            int64_t remain_data_len = (_send_flow_queue.send_flow[0].flow_size - _send_flow_queue.send_flow[0].snd_nxt);
            // if (remain_data_len <= 0 ) {continue;}

            int64_t payload_len = remain_data_len;
            if (remain_data_len > (MTU-header_len)) {   
                payload_len = MTU-header_len;
            }
            uint16_t pkt_size = payload_len+header_len;
            int64_t dequeue_time = _sim_time.sim_time + (pkt_size*8*1.0/_nic_rate.nic_rate)*(1000*1000*1000); // unit is ns
            int64_t enqueue_time = dequeue_time + _HS_link_delay.HS_link_delay;
            PktType pkt_type = PktType::DATA;
            
            int64_t sq_num = _send_flow_queue.send_flow[0].snd_nxt;

            Pkt pkt;
            create_pkt(pkt, _sender_id.sender_id, _send_flow_queue.send_flow[0].dst, _send_flow_queue.send_flow[0].l4_port, header_len, payload_len, header_len+payload_len, sq_num, flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type, path, 0, ECN_MARK::NO);

            #if PRINT_PKT_LOG
            PrintPkt(pkt, "data pkt sent by sender: ");
            #endif
            _enqueue(ctx.get<PktBuf>(in_port), pkt);

            _send_flow_queue.send_flow[0].snd_nxt += payload_len;

            // nxt_pkt_event must be >= dequeue_time, because cur_rate(m_rate) must be <= nic_rate;
            _send_flow_queue.send_flow[0].nxt_pkt_event = _sim_time.sim_time + (pkt_size*8*1.0/_send_flow_queue.send_flow[0].m_rate)*(1000*1000*1000);

            _sim_time.sim_time = _sim_time.sim_time > dequeue_time ? _sim_time.sim_time : dequeue_time;
            //if (pkt_cnt > 20) break;

            sent_bytes += pkt_size;
        }

    }

    // if (_sender_id.sender_id == 0 || _sender_id.sender_id == 1) {
    // if (_sender_id.sender_id == 0) {
        // printf("sender: %d, sim_time: %ld,, send rate is %.2f Gbps\n", _sender_id.sender_id, _sim_time.sim_time, sent_bytes*8*1.0/_sim_time_per_update.sim_time_per_update);
    // }

    _sim_time.sim_time = end_time;
    if (_sim_time.sim_time > _send_flow_queue.send_flow[0].nxt_pkt_event) {
        _send_flow_queue.send_flow[0].nxt_pkt_event = _sim_time.sim_time;
    }
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

inline void PrintPath(uint32_t *path, uint8_t path_len) {
    printf("path: length: %d", path_len);
    for(int32_t i = 0; i < path_len/2; i++) {
        // for (int32_t k = 2*i; k < 2 * (i+1); k++) {

        printf("(%d, %d), ", path[2*i], path[2*i+1]);
    }
    printf("\n");
    // for(int32_t i = 0; i < path_len; i++) {
    //     // for (int32_t k = 2*i; k < 2 * (i+1); k++) {
    //     printf("%d, ", path[i]);
    // }
    // printf("\n");
}

inline void set_forward_plan(Engine &ctx,
                            LocalPortID &_local_port_id,
                            GlobalPortID &_global_port_id,
                            SwitchID &_switch_id,
                            PktBuf &_queue,
                            ForwardPlan &_forward_plan,
                            SimTime & _sim_time,
                            SimTimePerUpdate &_sim_time_per_update)
{   
    // if (_global_port_id.global_port_id != 36) {
    //     return;
    // }
    if (_global_port_id.global_port_id == 0) {
        printf("\n\n*******Enter into set_forward_plan sys*********\n\n");
    }
    //reset
    for (uint8_t i = 0; i < INPORT_NUM; i++) {_forward_plan.forward_plan[i] = 0;}

    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;

    // printf("_switch_id: %d, _local_port_id: %d, global_port_id: %d\n", \
    //         _switch_id.switch_id, _local_port_id.local_port_id, _global_port_id.global_port_id);
    
    uint16_t q_len = get_queue_len(_queue);
    //if (q_len > 0 & _global_port_id.global_port_id == 0) PrintQueue(_queue);

    // if (_global_port_id.global_port_id == 0) {
    //     printf("set_forward_plan start sim_time: %lld\n", _sim_time.sim_time);
    //     printf("end_time: %lld\n", end_time);
    //     printf("the length of ingress queue: %d\n", q_len);
    //     printf("q_tail and q_head are: %d and %d\n", _queue.tail, _queue.head);
    // }
    // // if (_switch_id.switch_id == 8) {
    // printf("_switch_id: %d, _local_port_id: %d, global_port_id: %d\n", \
    //         _switch_id.switch_id, _local_port_id.local_port_id, _global_port_id.global_port_id);
    // // }
    uint16_t idx = 0;
    while(q_len--) { // at most processing q_len packets
        Pkt pkt;
        // if(idx > 20) break;
        // fetch_elem packet from buffer while checking whether buffer is empty
        if(!fetch_elem(_queue, idx, pkt)) break;

        if(pkt.enqueue_time >= end_time) break;
        // if (pkt.ecn == ECN_MARK::YES) {
        //PrintPkt(pkt, "forward queue: ");
        // }
        //printf("pkt.enqueue_time: %lld, end_time: %lld\n", pkt.enqueue_time, end_time);

        if (pkt.pkt_type != PktType::PFC_PAUSE && pkt.pkt_type != PktType::PFC_RESUME) {
            int32_t sw_idx = _switch_id.switch_id;
            //printf("sw_idx: %d, ", sw_idx);
            Entity sw_id = ctx.data()._switches[sw_idx];
            int32_t next_hop_link_idx = ctx.get<FIBTable>(sw_id).fib_table[pkt.dst][pkt.flow_id];
            _forward_plan.forward_plan[next_hop_link_idx] = _forward_plan.forward_plan[next_hop_link_idx] | (1<<idx);
        }
        //  broadcast PFC pause/resume frame into all egress port excepting for the current port itself
        else { 
            if (pkt.dst == 0) {// the pfc frame is received from the downstream egress port 
                Entity e_port = ctx.data().ePorts[_global_port_id.global_port_id]; 
                // the idx of priority queue is placed in the filed of pkt.flow_id
                // the PFC state is placed in the filed of the pkt.pkt_type
                if (pkt.pkt_type == PktType::PFC_PAUSE)
                    ctx.get<PktQueue>(e_port).queue_pfc_state[pkt.flow_id] = PFCState::PAUSE;
                else 
                    ctx.get<PktQueue>(e_port).queue_pfc_state[pkt.flow_id] = PFCState::RESUME;
            }

            for (int i = 0; i < INPORT_NUM; i++) { // broadcast
                if (i != _local_port_id.local_port_id) 
                    _forward_plan.forward_plan[i] = _forward_plan.forward_plan[i] | (1<<idx);
            }
        }

        idx += 1;
    }
    // printf("switch_id %d, local_port_id: %d\n", _switch_id.switch_id,_local_port_id.local_port_id);
    // if (_switch_id.switch_id == 1 && _local_port_id.local_port_id == 2) {
    //     printf("Forward plan of the %d-th ingress port in %d-th switch: \n", _local_port_id.local_port_id, _switch_id.switch_id);
    //     for(CountT i = 0; i < INPORT_NUM; i++) {
    //         printf("towards %d-th egress port: ", i);
    //         printBits<uint16_t>(_forward_plan.forward_plan[i]);
    //     }
    //     printf("\n");
    // }
}


// on *egress* ports
// each egress port actively fetches packets from in_ports
inline void _forward(Engine &ctx,
                    SchedTrajType sched_traj_type,
                    LocalPortID &_local_port_id,
                    GlobalPortID &_global_port_id,
                    PktQueue &_queue,
                    SwitchID &_switch_id,
                    SimTime &_sim_time,
                    SimTimePerUpdate &_sim_time_per_update)
{
    if (_global_port_id.global_port_id == 0) {
        printf("*******Enter into _forward sys*********\n");
    }
    // if (_global_port_id.global_port_id == 5) {
    //     return;
    // }
    Entity in_port;
    uint16_t fwd_bitmap;
    // int32_t _numInPort = ctx.data().numInPort;
  
    // FIXME: modify it to support the any shape of network topo
    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    // coupled with fattree topology
    for(CountT i = _switch_id.switch_id*K_ARY; i < (_switch_id.switch_id+1)*K_ARY; i++) { 

        in_port = ctx.data().inPorts[i];
        fwd_bitmap = ctx.get<ForwardPlan>(in_port).forward_plan[_local_port_id.local_port_id];

        PktBuf pkt_buf = ctx.get<PktBuf>(in_port);

        // if (_global_port_id.global_port_id == 5) {
        //     printf("_forward: \n");
        //     printBits<uint16_t>(fwd_bitmap);
        //     // PrintQueue(pkt_buf);
        // }

        //if (get_queue_len(_queue) <= 0) return;
        uint16_t num = 0;
        uint16_t count_bit = sizeof(fwd_bitmap)*8;
        while(count_bit--) {
            if ((fwd_bitmap & (1<<num)) != 0) { // the packet should be forward to the e_port
                // printf("the %d-th packet needs to be forward\n", num);
                Pkt pkt;
                //assert(_dequeue(pkt_buf, pkt) != 0);
                // PrintQueue(pkt_buf);
                fetch_elem(pkt_buf, num, pkt);
                //PrintPkt(pkt, "egress port pull pkts from ingress(forward): ");
                // path tracer
                pkt.path[pkt.path_len] = ctx.get<SwitchID>(in_port).switch_id;
                pkt.path[pkt.path_len+1] = ctx.get<LocalPortID>(in_port).local_port_id;
                pkt.path_len += 2;
                
                if (pkt.pkt_type == PktType::PFC_PAUSE || pkt.pkt_type == PktType::PFC_RESUME)  {
                    pkt.dst = 0;
                }

                // printf("%d", pkt.flow_id); 
                uint8_t queue_idx = pkt.flow_priority;
                _enqueue(_queue.pkt_buf[queue_idx], pkt);
                
            }
            num += 1;

            // if(_global_port_id.global_port_id==5 && i == 2) {
            //     printf("count_bit: %d\n", count_bit);
            // }
        }
    }
    //???????
    //_sim_time.sim_time = end_time; // FIXME
}


// on *ingress* port
// remove batch packets from ingress port buffer
inline void remove_pkts(Engine &ctx,
                        LocalPortID &_local_port_id,
                        GlobalPortID &_global_port_id,
                        PktBuf &_queue,
                        ForwardPlan &_forward_plan,
                        SimTime &_sim_time,
                        SimTimePerUpdate &_sim_time_per_update)
{   

    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    if (_global_port_id.global_port_id == 0) {
        printf("*******Enter into remove_pkts sys*********\n");
        // printf("end_time: %lld\n", end_time);
    }


    uint16_t q_len = get_queue_len(_queue);
    // if (_global_port_id.global_port_id == 6) {
    //     printf("global_port_id: %d, pkt buffer length before remove: %d\n", _global_port_id.global_port_id, q_len);
    // }
    // if (q_len <= 0) return;
    if (q_len <= 0) {
        _sim_time.sim_time = end_time;
        return;
    }

    // uint16_t idx = 0;
    Pkt pkt;
    while(fetch_elem(_queue, 0, pkt)) {
        // fetch_elem packet from buffer while checking whether buffer is empty
        //printf("enqueue_time: %ld, end_time: %ld\n", pkt.enqueue_time, end_time);
        if(pkt.enqueue_time >= end_time) {
            //printf("enqueue_time: %ld, end_time: %ld\n", pkt.enqueue_time, end_time);
            break;
        }
        _dequeue(_queue, pkt);
        //PrintPkt(pkt, "remove pkt in ingress port: ");
    }
    // if (_global_port_id.global_port_id == 6) {
    //     printf("global_port_id: %d, pkt buffer length after remove: %d\n", _global_port_id.global_port_id, get_queue_len(_queue));
    // }
    //???????
    _sim_time.sim_time = end_time; // FIXME
    
    //if (_global_port_id.global_port_id == 0) PrintQueue(_queue);

    // //reset forward_plan
    // for (uint8_t i = 0; i < INPORT_NUM; i++) {_forward_plan.forward_plan[i] = 0;}
}
//========================================================================================================


void transmit(Engine &ctx,
              SchedTrajType sched_traj_type, PortType port_type,
              LocalPortID &_local_port_id, GlobalPortID &_global_port_id,
              SwitchID &_switch_id, NextHop &_next_hop,
              NextHopType next_hop_type, PktQueue &_pkt_queue,
              TXHistory &_tx_history, QLen &_q_len, SSLinkDelay &_ss_link_delay,
              LinkRate &_link_rate, SimTime &_sim_time,
              SimTimePerUpdate &_sim_time_per_update, Seed &_seed)
{
    auto insertionSort = [](PktBuf &data) {
        int32_t i, j;
        Pkt key;
        uint16_t len = get_queue_len(data);
        for (i = 1; i < len; i++) {
            key = get_elem(data, i);
            j = i - 1;

            // Move elements of data[0..i-1] that are greater than key.enqueue_time to one position ahead of their current position
            while (j >= 0 && get_elem(data, j).enqueue_time > key.enqueue_time) {
                get_elem(data, j+1) = get_elem(data, j);

                j = j - 1;
            }
            get_elem(data, j+1) = key;
        }
    };

    if (_global_port_id.global_port_id == 0) {
        printf("*******Enter into transmit sys*********\n");
    }

    // if (_global_port_id.global_port_id != 2 && _global_port_id.global_port_id != 33) {
    //      return;
    // }

    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;

    uint8_t queue_num = ctx.get<QueueNumPerPort>(ctx.data()._switches[_switch_id.switch_id]).queue_num_per_port;
    // strict priority(SP) to schedule packets 
    //printf("queue_num: %d\n", queue_num);
    for (uint8_t i = 0; i < queue_num; i++) {
        if (get_queue_len(_pkt_queue.pkt_buf[i]) <= 0) continue;

        if (_pkt_queue.queue_pfc_state[i] == PFCState::PAUSE) {
            printf("transmit PFC state is PAUSE\n");
            continue;
        }

        // if (_global_port_id.global_port_id == 2) PrintQueue(_pkt_queue.pkt_buf[i]);
        // printf("before insertionSort: \n");
        insertionSort(_pkt_queue.pkt_buf[i]); // Sorting the packets in the queue using Insertion Sort
        // printf("after insertionSort: \n");

        // if (_global_port_id.global_port_id == 2) PrintQueue(_pkt_queue.pkt_buf[i]);
        
        // uint16_t pkt_cnt = 0;
        while(_sim_time.sim_time < end_time) {
            // if (_sim_time.sim_time >= end_time) break;
            // fetch_elem packet from buffer, while checking whether buffer is empty
            Pkt pkt;
            // printf("_dequeue _pkt_queue.pkt_buf[%d]: \n", i);

            if(!fetch_elem(_pkt_queue.pkt_buf[i], 0, pkt)) break;
            if(pkt.enqueue_time >= end_time) break;
            _dequeue(_pkt_queue.pkt_buf[i], pkt);

            uint32_t now_queue_len = 0; // record the queue length when a packet comes in real time
            int loc;  // FIXME
            for (loc = get_queue_len(_tx_history); loc > 0; loc--) {
                TXElem tx_elem = {0, 0};
                fetch_elem(_tx_history, loc-1, tx_elem);
                // printf("tx_elem.dequeue_time: %ld\n", tx_elem.dequeue_time);
                if (pkt.enqueue_time < tx_elem.dequeue_time) {
                    now_queue_len += tx_elem.ip_pkt_len;
                }
                else {
                    break;
                }
            }
            // // uint32_t now_queue_len = 0; // record the queue length when a packet comes in real time
            // // int loc;  // FIXME
            // uint16_t len = get_queue_len(_tx_history);
            // for (loc = 0; loc < len; loc++) {
            //     TXElem tx_elem = {0, 0};
            //     fetch_elem(_tx_history, loc, tx_elem);
            //     // printf("tx_elem.dequeue_time: %ld\n", tx_elem.dequeue_time);
            //     if (pkt.enqueue_time < tx_elem.dequeue_time) {
            //         now_queue_len += tx_elem.ip_pkt_len;
            //     }
            //     else {
            //         break;
            //     }
            // }
            
            // printf("global port id: %d, now the queue len: %d\n", _global_port_id.global_port_id, now_queue_len);
            // printf("clear tx_history: \n");

            if (loc > 0) {
                TXElem tmp_tx_elem;
                rm_queue_range(_tx_history, tmp_tx_elem, 0, loc); // FIXME, only handle when using 1 priority queue
            }

            // tail drop
            if ((pkt.payload_len+now_queue_len) > BUF_SIZE) {
                continue;
            }

            //uint16_t pkt_size = pkt.header_len + pkt.payload_len;
            pkt.dequeue_time = (pkt.enqueue_time > _sim_time.sim_time ? pkt.enqueue_time : _sim_time.sim_time) + \
                               (pkt.ip_pkt_len*8*1.0/_link_rate.link_rate)*(1000*1000*1000);
            pkt.enqueue_time = pkt.dequeue_time + _ss_link_delay.SS_link_delay; //FIXME

            TXElem tx_elem = {pkt.dequeue_time, pkt.ip_pkt_len};
            _enqueue(_tx_history, tx_elem); 

            // path tracer
            pkt.path[pkt.path_len] = _switch_id.switch_id;
            pkt.path[pkt.path_len+1] = _local_port_id.local_port_id;
            pkt.path_len += 2;
            
            //move packet from this egress port to next hop port / the dest host
            Entity next_hop_ett;
            if (next_hop_type == NextHopType::SWITCH) {
                next_hop_ett = ctx.data().inPorts[_next_hop.next_hop];
            }
            else {
                next_hop_ett = ctx.data().receivers[_next_hop.next_hop];
            }
            // if (_global_port_id.global_port_id == 4) continue;
            //
            // if (pkt.pkt_type == PktType::ACK || pkt.pkt_type == PktType::NACK) {
            //     printf("pkt_type: %d, switch_id: %d, global_port_id: %d, NextHopType: %d, next hop link: %d, src: %d --> dst: %d, payload_len: %d, sq_num: %lld, dequeue_time: %lld, enqueue_time: %lld\n", \
            //             pkt.pkt_type, _switch_id.switch_id, _global_port_id.global_port_id, next_hop_type, _next_hop.next_hop, pkt.src, pkt.dst, pkt.payload_len, pkt.sq_num, pkt.dequeue_time, pkt.enqueue_time);
            // }
            // //


            // ECN marking  
            // pkt.ecn = ECN_MARK::NO;
            // Create a LCG with parameters: a = 48271, c = 0, modulus = 2147483647, seed = 1
            LCG lcg(48271, 0, 2147483647, _seed.seed);
            float mark_prob;
            float rand_num;
            if (now_queue_len <= K_MIN && pkt.ecn == ECN_MARK::NO) {
                pkt.ecn = ECN_MARK::NO;
            }
            else if (now_queue_len > K_MIN && now_queue_len < K_MAX) {
                mark_prob = (now_queue_len-K_MIN)*P_MAX/(K_MAX-K_MIN);
                rand_num = (lcg.next()%1000)/1000.0;
                // printf("rand num is : %f\n", rand_num);
                if (rand_num < mark_prob) {pkt.ecn = ECN_MARK::YES;}
                _seed.seed = lcg.next();
            }
            else {
                pkt.ecn = ECN_MARK::YES;
            }

            // if (pkt.ecn == ECN_MARK::YES) {
                //printf("pkt.ecn: %d, rand_num: %f, mark_prob: %f\n", pkt.ecn, rand_num, mark_prob);
            //PrintPkt(pkt, "DATA pkt is marked with ecn: ");
            // }
            // float rand_num = (lcg.next()%1000)/1000.0;
            // printf("rand num is : %f\n", rand_num);

            _enqueue(ctx.get<PktBuf>(next_hop_ett), pkt);
            // if (pkt_cnt++ < 7) {
            //     _enqueue(ctx.get<PktBuf>(next_hop_ett), pkt);
            //     // pkt_cnt = 0;
            // }
            // _seed.seed = lcg.next()%1000;
            //printf("enqueue normal pkt: \n");

            _sim_time.sim_time = pkt.dequeue_time;

            // check whether the queue length excceeds the buffer limit 
            if (now_queue_len >= BUF_SIZE*(PFC_THR/100.0)) {
                uint32_t PFC_FOR_UPSTREAM = 1;
                int64_t pfc_enqueue_time = pkt.enqueue_time;
                uint8_t queue_idx = i; // pkt.flow_priority; // the idx of priority queue is placed in the filed(4-th element) of pkt
                uint8_t flow_priority = 0; // and the pfc frame is set with highest flow priority

                uint32_t path[MAX_PATH_LEN];
                memset(path, 0, MAX_PATH_LEN*sizeof(MAX_PATH_LEN));
                Pkt pfc_frame;
                // Pkt pfc_frame = {0, PFC_FOR_UPSTREAM, pkt.l4_port, 40, 0, 0, queue_idx, pfc_enqueue_time, 0, flow_priority, PktType::PFC_PAUSE}; // and PktType is PFC PAUSE
                printf("create_pkt: \n");

                create_pkt(pfc_frame, 0, PFC_FOR_UPSTREAM, pkt.l4_port, 40, 0, 40, 0, queue_idx, pfc_enqueue_time, 0, flow_priority, PktType::PFC_PAUSE, path, 0, ECN_MARK::NO);
                printf("after create_pkt: \n");
                Entity in_port_same = ctx.data().inPorts[_global_port_id.global_port_id]; // in_port in the same port as the egress port
                printf("enqueue pfc_frame: \n");
                _enqueue(ctx.get<PktBuf>(in_port_same), pfc_frame);
                printf("after enqueue pfc_frame: \n"); 
            }
        }
    }
    //???????
    _sim_time.sim_time = end_time; // FIXME
}


// receive system: 
// 1) generate ACK/NACK/CNP; 
// 2) or move them into the ack buffer of receiver's sender 
inline void receive(Engine &ctx, ReceiverID &_receiver_id, NextHop &_next_hop, PktBuf &_queue,\
                    NICRate &_nic_rate, HSLinkDelay &_HS_link_delay, SimTime &_sim_time, \
                    SimTimePerUpdate &_sim_time_per_update, RecvFlowQueue &recv_flow_queue) {

    int64_t end_time = _sim_time.sim_time + _HS_link_delay.HS_link_delay;
    if (_receiver_id.receiver_id == 0) {
        printf("*******Enter into receive system*********\n");
        printf("start _sim_time.sim_time: %ld\n", _sim_time.sim_time);
        printf("end_time: %ld\n", end_time);
    }

    Pkt pkt;
    uint8_t flow_priority = 0;
    uint16_t header_len = 40;
    while (_sim_time.sim_time < end_time) {
        if(!fetch_elem(_queue, 0, pkt)) break;
        if(pkt.enqueue_time >= end_time) break;
        _dequeue(_queue, pkt);

        // PrintPath(pkt.path, pkt.path_len);
        if(pkt.enqueue_time >= end_time) break;
        
        if (pkt.pkt_type != PktType::DATA) { // ACK/NACK/CNP pkt
            Entity sender_ett = ctx.data().senders[_receiver_id.receiver_id];
            _enqueue(ctx.get<PktBuf>(sender_ett), pkt); //enqueue into the ack_pkt_buf

            // printf("receiver_id: %d\n", _receiver_id.receiver_id);
            #if PRINT_PKT_LOG
            PrintPkt(pkt, "ACK/NACK received by sender's receiver: ");
            #endif
        }
        else { //DATA pkt
            uint16_t pkt_size = pkt.header_len + pkt.payload_len;
            _sim_time.sim_time += (pkt_size*8*1.0/_nic_rate.nic_rate)*(1000*1000*1000);

            // printf("receiver_id: %d\n", _receiver_id.receiver_id);
            // printf("src: %d recv: %d, expected_recv_bytes: %ld\n", recv_flow_queue.recv_flow[0].src, recv_flow_queue.recv_flow[0].dst, recv_flow_queue.recv_flow[0].recv_bytes);
            #if PRINT_PKT_LOG
            PrintPkt(pkt, "DATA received by receiver: ");
            #endif
            uint32_t path[MAX_PATH_LEN];
            memset(path, 0, MAX_PATH_LEN*sizeof(MAX_PATH_LEN));

            // int64_t expect_byte = + payload_len;
            uint32_t ack_src = pkt.dst; //recv_flow_queue.recv_flow[0].dst;
            uint32_t ack_dst = pkt.src; //recv_flow_queue.recv_flow[0].src;

            int64_t dequeue_time = _sim_time.sim_time + 1;
            int64_t enqueue_time = _sim_time.sim_time + 1 + _HS_link_delay.HS_link_delay;
            
            int64_t sq_num;
            Pkt ack_pkt;
            if (pkt.sq_num == recv_flow_queue.recv_flow[0].recv_bytes) { // packet order is right: generate ACK
                recv_flow_queue.recv_flow[0].recv_bytes += pkt.payload_len;
                sq_num = recv_flow_queue.recv_flow[0].recv_bytes;
                PktType pkt_type = PktType::ACK;
                // int64_t dequeue_time = _sim_time.sim_time + 1;
                // int64_t enqueue_time = _sim_time.sim_time + 1 + _HS_link_delay.HS_link_delay;

                // Pkt ack_pkt = {recv_flow_queue.recv_flow[0].src, recv_flow_queue.recv_flow[0].dst, recv_flow_queue.recv_flow[0].l4_port, header_len, 0, sq_num, pkt.flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type};
                create_pkt(ack_pkt, ack_src, ack_dst, \
                        recv_flow_queue.recv_flow[0].l4_port, header_len, 0, header_len, sq_num, \
                        pkt.flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type, path, 0, ECN_MARK::NO);
            }
            else { // packets is out of order, NACK
                sq_num = recv_flow_queue.recv_flow[0].recv_bytes;
                PktType pkt_type = PktType::NACK;
                // int64_t dequeue_time = _sim_time.sim_time + 1;
                // int64_t enqueue_time = _sim_time.sim_time + 1 + _HS_link_delay.HS_link_delay;

                // Pkt ack_pkt = {recv_flow_queue.recv_flow[0].src, recv_flow_queue.recv_flow[0].dst, recv_flow_queue.recv_flow[0].l4_port, header_len, 0, sq_num, pkt.flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type};
                create_pkt(ack_pkt, ack_src, ack_dst, \
                           recv_flow_queue.recv_flow[0].l4_port, header_len, 0, header_len, sq_num, \
                           pkt.flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type, path, 0, ECN_MARK::NO);     
            }
            
            Entity direct_connect_in_port = ctx.data().inPorts[_next_hop.next_hop];
            _enqueue(ctx.get<PktBuf>(direct_connect_in_port), ack_pkt);
            #if PRINT_PKT_LOG
            PrintPkt(ack_pkt, "ACK/NACK generated by receiver: ");
            #endif
            // printf("pkt.ecn: %d\n", pkt.ecn);
            // packet with ecn marking, generate a cnp packet
            if (pkt.ecn == ECN_MARK::YES && (_sim_time.sim_time - recv_flow_queue.recv_flow[0].last_cnp_timestamp) >= CNP_DURATION) { 
            // if (pkt.ecn == ECN_MARK::YES) { 
                Pkt cnp_pkt;
                PktType pkt_type = PktType::CNP;
                create_pkt(cnp_pkt, ack_src, ack_dst, \
                           recv_flow_queue.recv_flow[0].l4_port, header_len, 0, header_len, sq_num, \
                           pkt.flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type, path, 0, ECN_MARK::YES);
                            // Entity direct_connect_in_port = ctx.data().inPorts[_next_hop.next_hop];
                _enqueue(ctx.get<PktBuf>(direct_connect_in_port), cnp_pkt);
                #if PRINT_PKT_LOG
                PrintPkt(cnp_pkt, "CNP generated by receiver: ");
                #endif
                recv_flow_queue.recv_flow[0].last_cnp_timestamp = _sim_time.sim_time;
            }
        }
        //recv_flow_queue[0].ack_sq_num += packet.payload_len;
    }
    _sim_time.sim_time = end_time;
}



inline void _debug_e_port(Engine &ctx,
                        PortType port_type,
                        GlobalPortID &_global_port_id,
                        PktBuf &_queue)
{
    
}



void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
{
    printf("*******Enter into setupTasks*********\n");

    auto receive_sys = builder.addToGraph<ParallelForNode<Engine, receive, ReceiverID, NextHop, PktBuf, \
                                           NICRate, HSLinkDelay, SimTime, SimTimePerUpdate, RecvFlowQueue>>({}); 
    auto send_sys = builder.addToGraph<ParallelForNode<Engine, send, SenderID, NextHop, NICRate, \
                                       HSLinkDelay, SimTime, SimTimePerUpdate, SendFlowQueue, PktBuf>>({receive_sys});

    auto set_forward_plan_sys = builder.addToGraph<ParallelForNode<Engine, set_forward_plan, LocalPortID, GlobalPortID, \
                                                   SwitchID, PktBuf, ForwardPlan, SimTime, SimTimePerUpdate>>({send_sys});

    auto forward_sys = builder.addToGraph<ParallelForNode<Engine, _forward, SchedTrajType, \
                                          LocalPortID, GlobalPortID, PktQueue, SwitchID, \
                                          SimTime, SimTimePerUpdate>>({set_forward_plan_sys});

    auto remove_pkts_sys = builder.addToGraph<ParallelForNode<Engine, remove_pkts, \
                                              LocalPortID, GlobalPortID, PktBuf, ForwardPlan, \
                                              SimTime, SimTimePerUpdate>>({forward_sys}); 

    auto transmit_sys = builder.addToGraph<ParallelForNode<Engine, transmit, SchedTrajType, \
                                           PortType, LocalPortID,  GlobalPortID, SwitchID, \
                                           NextHop, NextHopType, PktQueue, TXHistory, QLen, \
                                           SSLinkDelay, LinkRate, SimTime, \
                                           SimTimePerUpdate, Seed>>({remove_pkts_sys}); 

    // auto receive_sys = builder.addToGraph<ParallelForNode<Engine, receive, ReceiverID, NextHop, PktBuf, \
    //                                       NICRate, HSLinkDelay, SimTime, SimTimePerUpdate, RecvFlowQueue>>({transmit_sys}); 

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
