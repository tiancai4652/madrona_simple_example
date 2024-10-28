#include "2_sim_mflow.hpp"
// #include "dcqcn.hpp"
#include "2_build_topo_mflow.hpp"
// #include "myQueue.hpp"

#include <madrona/mw_gpu_entry.hpp> 
#include <atomic>

// for conditional compilination
#include <type_traits>
#include <utility>

using namespace madrona;
using namespace madrona::math;

#define PRINT_PKT_LOG 0

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
    registry.registerComponent<Results>();
    registry.registerComponent<Results2>();
    registry.registerComponent<SimulationTime>();
    registry.registerComponent<MadronaEventsQueue>();
    registry.registerComponent<MadronaEvents>();
    registry.registerComponent<MadronaEventsResult>();
    registry.registerComponent<ProcessParams>();
    registry.registerArchetype<Agent>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
    registry.exportColumn<Agent, Results>((uint32_t)ExportID::Results);
    registry.exportColumn<Agent, Results2>((uint32_t)ExportID::Results2);
    registry.exportColumn<Agent, SimulationTime>((uint32_t)ExportID::SimulationTime);
    registry.exportColumn<Agent, MadronaEvents>((uint32_t)ExportID::MadronaEvents);
    registry.exportColumn<Agent, MadronaEventsResult>((uint32_t)ExportID::MadronaEventsResult);
    registry.exportColumn<Agent, ProcessParams>((uint32_t)ExportID::ProcessParams);


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
    //PktQueue is composed of multiple PktBuf and a PFCState array
    registry.registerComponent<PktQueue>(); 
    
    registry.registerComponent<SimTime>();
    registry.registerComponent<SimTimePerUpdate>();


    ///////////////////////////////////////////////////////
    // components for send flow entity
    registry.registerComponent<FlowID>();
    registry.registerComponent<Src>();
    registry.registerComponent<Dst>();
    registry.registerComponent<L4Port>();

    registry.registerComponent<FlowSize>();
    registry.registerComponent<StartTime>();
    registry.registerComponent<StopTime>();

    registry.registerComponent<NIC_ID>();
    registry.registerComponent<SndServerID>();
    registry.registerComponent<RecvServerID>();
    
    registry.registerComponent<SndNxt>();
    registry.registerComponent<SndUna>();
    registry.registerComponent<FlowState>();
    registry.registerComponent<Extra_1>();
    
    registry.registerComponent<AckPktBuf>();

    registry.registerComponent<LastAckTimestamp>();
    registry.registerComponent<NxtPktEvent>();
    registry.registerComponent<CC_Para>();

    registry.registerArchetype<SndFlow>();

    ///////////////////////////////////////////////////////
    // components for recv flow entity
    registry.registerComponent<LastCNPTimestamp>();
    registry.registerComponent<RecvBytes>();

    registry.registerArchetype<RecvFlow>();
    
    registry.registerComponent<NPU_ID>();
    registry.registerComponent<SendFlows>();
    registry.registerComponent<NewFlowQueue>(); 
    registry.registerComponent<CompletedFlowQueue>();



    registry.registerArchetype<NPU>();

    ///////////////////////////////////////////////////////
    // components for NIC entity
    registry.registerComponent<MountedFlows>();
    registry.registerComponent<BidPktBuf>();
    registry.registerComponent<TXHistory>();  
    registry.registerComponent<Seed>();

    registry.registerArchetype<NIC>();

    ///////////////////////////////////////////////////////
    // components for switch entity
    registry.registerComponent<SwitchID>();
    registry.registerComponent<SwitchType>();
    registry.registerComponent<FIBTable>();
    registry.registerComponent<QueueNumPerPort>();
    
    registry.registerArchetype<Switch>();

    ///////////////////////////////////////////////////////
    // components for egress and ingress entity
    registry.registerComponent<LinkRate>();
    registry.registerComponent<SSLinkDelay>();

    registry.registerComponent<SchedTrajType>();
    registry.registerComponent<PortType>();
    registry.registerComponent<LocalPortID>();
    registry.registerComponent<GlobalPortID>();

    registry.registerComponent<ForwardPlan>();

    registry.registerArchetype<IngressPort>();
    registry.registerArchetype<EgressPort>();

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
inline bool _dequeue(T1& queue, T2& elem) 
{
    if (queue.cur_num == 0) {
        // The queue is empty, cannot dequeue
        return false;
    }
    elem = queue.pkts[queue.head];
    queue.head = (queue.head + 1) % PKT_BUF_LEN;
    queue.cur_num -= 1;

    if constexpr (has_cur_bytes<T1>::value) {
        queue.cur_bytes -= elem.ip_pkt_len;
    }
    // queue.cur_bytes -= elem.ip_pkt_len;

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
    if (queue.cur_num >= PKT_BUF_LEN) {
        printf("queue is overloaded\n");
        // exit(0);
    }

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


template<typename T1, typename T2>
inline bool _dequeue_flow(T1& queue, T2& elem) 
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
inline void _enqueue_flow(T1 &queue, T2 pkt) {
    if (queue.cur_num >= MAX_NPU_FLOW_NUM) {
        printf("error: flow queue is overloaded\n");
        // exit(0);
    }
    queue.flow[queue.tail] = pkt;
    queue.tail = (queue.tail + 1) % MAX_NPU_FLOW_NUM;
    queue.cur_num += 1;
}


// template<typename T1, typename T2>
// inline bool _dequeue_entt(T1& queue, T2& elem) 
// {
//     if (queue.cur_num == 0) {
//         // The queue is empty, cannot dequeue
//         return false;
//     }

//     elem = queue.flow[queue.head];
//     queue.head = (queue.head + 1) % MAX_NPU_FLOW_NUM;
//     queue.cur_num -= 1;

//     return true; 
// }


// template<typename T1, typename T2>
// inline void _enqueue_entt(T1 &queue, T2 pkt) {
//     if (queue.cur_num >= MAX_NPU_FLOW_NUM) {
//         printf("error: flow queue is overloaded\n");
//         exit(0);
//     }
//     queue.flow[queue.tail] = pkt;
//     queue.tail = (queue.tail + 1) % MAX_NPU_FLOW_NUM;
//     queue.cur_num += 1;
// }


///////////////////////////////////////////////////////////////////////////////////
// DCQCN CC logic
void inline dcqcn_rate_increase(CC_Para &cc_para, uint32_t src, long curSimTime)
{
    if (cc_para.dcqcn_IncreaseStageCount < cc_para.dcqcn_RecoveryStageThreshold)
    {
        // fast recovery stage
        // update current rate
        cc_para.m_rate = cc_para.m_rate / 2 + cc_para.tar_rate / 2;

        printf("In fast recovery stage. sender: %d, sim_time: %ld, target rate is %.3lf Gbps, and current rate is %.3lf Gbps\n", src, curSimTime, cc_para.tar_rate/(1000*1000*1000.0), cc_para.m_rate/(1000*1000*1000.0));
    }
    else if (cc_para.dcqcn_IncreaseStageCount == cc_para.dcqcn_RecoveryStageThreshold)
    {
        // additive increase stage
        // target rate increases with fixed AI sending rate
        cc_para.tar_rate += cc_para.dcqcn_AIRate;

        // if target rate is bigger than the device rate, set to device rate
        if (cc_para.tar_rate > NIC_RATE) {
            cc_para.tar_rate = NIC_RATE;
        }
        // update current rate
        cc_para.m_rate = cc_para.m_rate / 2 + cc_para.tar_rate / 2;

        // printf("In additive increase stage. sender: %d, sim_time: %ld, target rate is %.3lf Gbps, and current rate is %.3lf Gbps\n", src, curSimTime, cc_para.tar_rate/(1000*1000*1000.0), cc_para.m_rate/(1000*1000*1000.0));
    }
    else
    {
        // hyper increase stage
        // target rate increases with fixed AI sending rate
        cc_para.tar_rate += cc_para.dcqcn_HAIRate;

        // if target rate is bigger than the device rate, set to device rate
        if (cc_para.tar_rate > NIC_RATE) {
            cc_para.tar_rate = NIC_RATE;
        }
        // update current rate
        cc_para.m_rate = cc_para.m_rate / 2 + cc_para.tar_rate / 2;

        printf("In hyper increase stage. sender: %d, sim_time: %ld, target rate is %.3lf Gbps, and current rate is %.3lf Gbps\n", src, curSimTime, cc_para.tar_rate/(1000*1000*1000.0), cc_para.m_rate/(1000*1000*1000.0));
    }

    // increase the stage count
    cc_para.dcqcn_IncreaseStageCount++;
}

void inline dcqcn_alpha_update(CC_Para &cc_para)
{
    // if we receive CNP packet, increase value of alpha
    if (cc_para.CNPState == true)
    {
        cc_para.dcqcn_Alpha = (1 - cc_para.dcqcn_G) * cc_para.dcqcn_Alpha + cc_para.dcqcn_G;
        // cc_para.CNPState = false;
    }
    // else decrease the value of alpha
    else
    {
        cc_para.dcqcn_Alpha = (1 - cc_para.dcqcn_G) * cc_para.dcqcn_Alpha;
    }
}

// decrease stage of sending rate in DCQCN, when received ECE packet
void inline dcqcn_rate_decrease(CC_Para &cc_para, uint32_t src, long curSimTime) {
    // if (cc_para.ECNState != ECNState.ECE_RECV)
    if (cc_para.CNPState != true)
        return;
    // record current rate to later fast recovery
    // printf(" original rate: %.3lf, dcqcn_Alpha: %.3lf\n", cc_para.m_rate/(1000*1000*1000.0), cc_para.dcqcn_Alpha);
    
    cc_para.tar_rate = cc_para.m_rate;
    cc_para.m_rate = (long int)(cc_para.m_rate * (1.0 - cc_para.dcqcn_Alpha / 2.0));
    cc_para.dcqcn_IncreaseStageCount = 0;

    printf("In decrease stage. sender: %d, sim_time: %ld, target rate is %.3lf Gbps, and current rate is %.3lf Gbps\n", src, curSimTime, cc_para.tar_rate/(1000*1000*1000.0), cc_para.m_rate/(1000*1000*1000.0));
}

// reset the alpha update and rate increase timer, when rate is decreased
void inline dcqcn_timer_reset_if_decrease(CC_Para &cc_para, long curSimTime) {

    // if rate isn't decreased
    if (!cc_para.dcqcn_RateIsDecreased)
        return;

    // since we decrease the rate, we need to reset rate increase timer
    cc_para.dcqcn_AlphaUpdateNextTime = cc_para.dcqcn_RateIncreaseNextTime = curSimTime + cc_para.dcqcn_RateIncreaseInterval;
    //
    cc_para.dcqcn_RateIsDecreased = false;
}

// 
void inline dcqcn_timer_rate_increase(CC_Para &cc_para, uint32_t src, long curSimTime) {
    // if current simulation time is smaller than the next rate increase time,
    // should not tigger this function
    if (curSimTime <= cc_para.dcqcn_RateIncreaseNextTime)
        return;

    // schedule the next rate increase timer
    cc_para.dcqcn_RateIncreaseNextTime += cc_para.dcqcn_RateIncreaseInterval;

    // increase sending rate
    dcqcn_rate_increase(cc_para, src, curSimTime);

}

void inline dcqcn_timer_alpha_update(CC_Para &cc_para, long curSimTime) {

    // if current simulation time is smaller than the next rate decrease time,
    // should not tigger this function
    if (curSimTime <= cc_para.dcqcn_AlphaUpdateNextTime)
        return;
    // schedule the next alpha update timer
    cc_para.dcqcn_AlphaUpdateNextTime += cc_para.dcqcn_AlphaUpdateInterval;

    dcqcn_alpha_update(cc_para);
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


// inline void inject_comm(Engine &ctx, FlowEvent flow_event, Entity flow_entt) {
//     ctx.get<Src>(flow_entt).src = flow_event.src;
//     ctx.get<Dst>(flow_entt).dst = flow_event.dst;
//     ctx.get<L4Port>(flow_entt).l4_port = flow_event.l4_port;
//     ctx.get<NIC_ID>(flow_entt).nic_id = flow_event.nic;

//     ctx.get<FlowSize>(flow_entt).flow_size = flow_event.flow_size;
//     ctx.get<StartTime>(flow_entt).start_time = flow_event.start_time;
//     ctx.get<FlowState>(flow_entt) = FlowState::UNCOMPLETE;
//     // ctx.get<FlowState>(flow_entt).flow_state = FlowState::UNCOMPLETE;
// }

// _new_flow_queue.new_flow_queue is from astra-sim system layer
// inline void setup_flow(Engine &ctx, SendFlows &_flow_entts,
//                        NewFlowQueue & _new_flow_queue, SimTime &_sim_time,
//                        SimTimePerUpdate &_sim_time_per_update) {
//     uint32_t flow_event_num = get_queue_len(_new_flow_queue);
//     for (uint32_t i = 0; i < flow_event_num; i++)
//     {
//         FlowEvent flow_event;
//         _dequeue_flow(_new_flow_queue, flow_event);
//         Entity flow_entt = _flow_entts.flow[flow_event.flow_id];
//         inject_comm(ctx, flow_event, flow_entt); //or create flow entities
//     }
// }

// create send flow entity and recv flow entity
inline void create_flow(Engine &ctx, FlowEvent flow_event, uint32_t flow_id) {
// snd flows
    Entity snd_flow_entt = ctx.makeEntity<SndFlow>();

    ctx.get<FlowID>(snd_flow_entt).flow_id = flow_id;
    ctx.get<Src>(snd_flow_entt).src = flow_event.src;
    ctx.get<Dst>(snd_flow_entt).dst = flow_event.dst;
    ctx.get<L4Port>(snd_flow_entt).l4_port = flow_event.l4_port; //FIXME

    ctx.get<FlowSize>(snd_flow_entt).flow_size = flow_event.flow_size;
    ctx.get<StartTime>(snd_flow_entt).start_time = flow_event.start_time;
    ctx.get<StopTime>(snd_flow_entt).stop_time = 0;

    ctx.get<NIC_ID>(snd_flow_entt).nic_id = flow_event.src;
    // ctx.get<SndServerID>(snd_flow_entt).snd_server_id = flow_events[i][6];
    // ctx.get<RecvServerID>(snd_flow_entt).recv_server_id = flow_events[i][7];
    
    ctx.get<SndNxt>(snd_flow_entt).snd_nxt = 0;
    ctx.get<SndUna>(snd_flow_entt).snd_una = 0;
    ctx.get<FlowState>(snd_flow_entt) = FlowState::UNCOMPLETE;
    ctx.get<Extra_1>(snd_flow_entt).extra_1 =flow_event.extra_1;

    ctx.get<LastAckTimestamp>(snd_flow_entt).last_ack_timestamp = 0; //1ms
    ctx.get<NxtPktEvent>(snd_flow_entt).nxt_pkt_event = 0;
    
    ctx.get<PFCState>(snd_flow_entt) = PFCState::RESUME;

    ctx.get<CC_Para>(snd_flow_entt).m_rate = DCQCN_CurrentRate;
    ctx.get<CC_Para>(snd_flow_entt).tar_rate = DCQCN_TargetRate;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_Alpha = DCQCN_Alpha;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_G = DCQCN_G;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_AIRate = DCQCN_AIRate;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_HAIRate = DCQCN_HAIRate; 
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_RateDecreaseInterval = DCQCN_RateDecreaseInterval;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_RateIncreaseInterval = DCQCN_RateIncreaseInterval;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_AlphaUpdateInterval = DCQCN_AlphaUpdateInterval;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_IncreaseStageCount = DCQCN_IncreaseStageCount;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_RateDecreaseNextTime = DCQCN_RateDecreaseNextTime;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_RateIncreaseNextTime = DCQCN_RateIncreaseNextTime;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_AlphaUpdateNextTime = DCQCN_AlphaUpdateNextTime;
    ctx.get<CC_Para>(snd_flow_entt).dcqcn_RateIsDecreased = DCQCN_RateIsDecreased;
    ctx.get<CC_Para>(snd_flow_entt).CNPState = false;

    ctx.get<PktBuf>(snd_flow_entt).head = 0;
    ctx.get<PktBuf>(snd_flow_entt).tail = 0;
    ctx.get<PktBuf>(snd_flow_entt).cur_num = 0;
    ctx.get<PktBuf>(snd_flow_entt).cur_bytes = 0;

    ctx.get<AckPktBuf>(snd_flow_entt).head = 0;
    ctx.get<AckPktBuf>(snd_flow_entt).tail = 0;
    ctx.get<AckPktBuf>(snd_flow_entt).cur_num = 0;
    ctx.get<AckPktBuf>(snd_flow_entt).cur_bytes = 0;

    ctx.get<NICRate>(snd_flow_entt).nic_rate = 1000LL*1000*1000*100; // 100 Gbps
    ctx.get<HSLinkDelay>(snd_flow_entt).HS_link_delay = HS_LINK_DELAY; // 1 us, 1000 ns 

    ctx.get<SimTime>(snd_flow_entt).sim_time = 0;
    ctx.get<SimTimePerUpdate>(snd_flow_entt).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

    uint32_t src = flow_event.src;
    // mounted to NPU
    Entity npu_entt = ctx.data()._npus[src];
    _enqueue_entt(ctx.get<SendFlows>(npu_entt), snd_flow_entt);
    // mounted to NIC
    Entity nic_entt = ctx.data()._nics[ctx.get<NIC_ID>(snd_flow_entt).nic_id];
    _enqueue_entt(ctx.get<MountedFlows>(nic_entt), snd_flow_entt);

    ctx.data().snd_flows[flow_event.src][flow_id] = snd_flow_entt;
//

// recv flows
    Entity rcv_flow_entt = ctx.makeEntity<RecvFlow>();
    
    ctx.get<FlowID>(rcv_flow_entt).flow_id = flow_id;
    ctx.get<Src>(rcv_flow_entt).src = flow_event.src;
    ctx.get<Dst>(rcv_flow_entt).dst = flow_event.dst;
    ctx.get<L4Port>(rcv_flow_entt).l4_port = flow_event.l4_port; //FIXME

    ctx.get<FlowSize>(rcv_flow_entt).flow_size = flow_event.flow_size;
    ctx.get<StartTime>(rcv_flow_entt).start_time = flow_event.start_time;
    ctx.get<StopTime>(rcv_flow_entt).stop_time = 0;

    ctx.get<NIC_ID>(rcv_flow_entt).nic_id = flow_event.src;

    ctx.get<FlowState>(rcv_flow_entt) = FlowState::UNCOMPLETE;

    ctx.get<LastCNPTimestamp>(rcv_flow_entt).last_cnp_timestamp = 0; //1ms

    ctx.get<RecvBytes>(rcv_flow_entt).recv_bytes = 0;

    ctx.get<PktBuf>(rcv_flow_entt).head = 0;
    ctx.get<PktBuf>(rcv_flow_entt).tail = 0;
    ctx.get<PktBuf>(rcv_flow_entt).cur_num = 0;
    ctx.get<PktBuf>(rcv_flow_entt).cur_bytes = 0;

    ctx.get<NICRate>(rcv_flow_entt).nic_rate = 1000LL*1000*1000*100; // 100 Gbps  
    ctx.get<HSLinkDelay>(rcv_flow_entt).HS_link_delay = HS_LINK_DELAY; // 1 us, 1000 ns 

    ctx.get<SimTime>(rcv_flow_entt).sim_time = 0;
    ctx.get<SimTimePerUpdate>(rcv_flow_entt).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

    ctx.data().recv_flows[flow_event.src][flow_id] = rcv_flow_entt;
//
}


inline void setup_flow(Engine &ctx, NPU_ID _npu_id,
                       NewFlowQueue & _new_flow_queue, SimTime &_sim_time,
                       SimTimePerUpdate &_sim_time_per_update) {
    
    // if (_npu_id.npu_id == 0) {printf("*********Enter into setup_flow:*********\n");}

    uint32_t flow_event_num = get_queue_len(_new_flow_queue);
    for (uint32_t i = 0; i < flow_event_num; i++)
    {
        FlowEvent flow_event;
        _dequeue_flow(_new_flow_queue, flow_event);
        uint32_t flow_id = ctx.data().flow_count[_npu_id.npu_id];
        create_flow(ctx, flow_event, flow_id); //or create flow entities
    }

    clear_queue(_new_flow_queue);
}


// _completed_flow_queue is for astra-sim system layer
// inline void check_flow_state(Engine &ctx, SendFlows &_flow_entts,
//                              CompletedFlowQueue &_completed_flow_queue,
//                              SimTime &_sim_time, SimTimePerUpdate &_sim_time_per_update) {
//     uint32_t flow_num = get_queue_len(_flow_entts);
//     for (uint32_t i = 0; i < flow_num; i++) {
//         Entity flow_entt = _flow_entts.flow[i];
//         if (ctx.get<FlowState>(flow_entt) == FlowState::COMPLETE) {
//             FlowEvent flow_event = {ctx.get<FlowID>(flow_entt).flow_id, ctx.get<Src>(flow_entt).src, ctx.get<Dst>(flow_entt).dst, \
//                                     ctx.get<L4Port>(flow_entt).l4_port, ctx.get<NIC_ID>(flow_entt).nic_id, \
//                                     ctx.get<FlowSize>(flow_entt).flow_size, ctx.get<StartTime>(flow_entt).start_time, \
//                                     ctx.get<StopTime>(flow_entt).stop_time, ctx.get<FlowState>(flow_entt)};
//             _enqueue_flow(_completed_flow_queue, flow_event);
//         }
//     }
// }

inline void check_flow_state(Engine &ctx, NPU_ID &_npu_id, CompletedFlowQueue &_completed_flow_queue,
                             SimTime &_sim_time, SimTimePerUpdate &_sim_time_per_update) {

    uint32_t tmp_snd_flow[MAP_SIZE];
    memset(tmp_snd_flow, 0, MAX_PATH_LEN*sizeof(uint32_t));
    uint32_t cnt = 0; 

    Map<uint16_t, Entity> all_snd_flow = ctx.data().snd_flows[_npu_id.npu_id];
    
    Map<uint16_t, Entity> all_recv_flow = ctx.data().recv_flows[_npu_id.npu_id];
    
    for (auto it = all_snd_flow.begin(); it != all_snd_flow.end(); ++it) {
        auto& entry = *it;
        Entity flow_entt = entry.value;

        if (ctx.get<FlowState>(flow_entt) == FlowState::COMPLETE) {
            FlowEvent flow_event = {ctx.get<FlowID>(flow_entt).flow_id, ctx.get<Src>(flow_entt).src, ctx.get<Dst>(flow_entt).dst, \
                                    ctx.get<L4Port>(flow_entt).l4_port, ctx.get<NIC_ID>(flow_entt).nic_id, \
                                    ctx.get<FlowSize>(flow_entt).flow_size, ctx.get<StartTime>(flow_entt).start_time, \
                                    ctx.get<StopTime>(flow_entt).stop_time, ctx.get<FlowState>(flow_entt),ctx.get<Extra_1>(flow_entt).extra_1};
            _enqueue_flow(_completed_flow_queue, flow_event);
            //

            tmp_snd_flow[cnt++] = ctx.get<FlowID>(flow_entt).flow_id;

            Entity recv_flow_entt = all_recv_flow[ctx.get<FlowID>(flow_entt).flow_id];
            ctx.destroyEntity(recv_flow_entt);

            ctx.destroyEntity(flow_entt);
        }
    }

    for (uint32_t i = 0; i < cnt; i++) {
        ctx.data().snd_flows[_npu_id.npu_id].remove(tmp_snd_flow[i]);
    }
    for (uint32_t i = 0; i < cnt; i++) {
        ctx.data().recv_flows[_npu_id.npu_id].remove(tmp_snd_flow[i]);
    }

    clear_queue(_completed_flow_queue);
}




inline void flow_send(Engine &ctx, FlowID &_flow_id, Src &_src, Dst &_dst, L4Port &_l4_port, 
                      FlowSize &_flow_size, StartTime &_start_time, StopTime &_stop_time,
                      NIC_ID &_nic_id, SndServerID &_snd_server_id, RecvServerID &_recv_server_id,
                      SndNxt &_snd_nxt, SndUna &_snd_una, FlowState &_flow_state, 
                      LastAckTimestamp &_last_ack_timestamp, NxtPktEvent &_nxt_pkt_event, 
                      PFCState &_pfc_state, CC_Para &_cc_para, 
                      PktBuf &_snd_buf, AckPktBuf &_ack_buf, 
                      NICRate &_nic_rate, HSLinkDelay &_HS_link_delay, 
                      SimTime &_sim_time, SimTimePerUpdate &_sim_time_per_update) {
    // Entity in_port = ctx.data().inPorts[_next_hop.next_hop];
    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    // if (_flow_id.flow_id == 0 || _flow_id.flow_id == 1) {
    //     printf("\n*******Enter into flow send*********\n");
    //     printf("start _sim_time.sim_time: %ld\n", _sim_time.sim_time);
    //     printf("end_time: %ld\n", end_time);
    //     // printf("cc target rate: %.3lf, cc cur rate: %.3lf\n", _cc_para.tar_rate/(1000*1000*1000.0), _m_rate.m_rate/(1000*1000*1000.0));
    // }

    uint16_t flow_id = 0;
    uint16_t header_len = 40; 
    uint8_t flow_priority = 0;
    uint32_t path[MAX_PATH_LEN];
    memset(path, 0, MAX_PATH_LEN*sizeof(uint32_t));

    int16_t sent_bytes = 0; 
    while (_sim_time.sim_time < end_time) {
        //printf("send sim_time: %lld\n", _sim_time.sim_time);   
        //printf("_sim_time.sim_time < end_time");
        Pkt ack_p;
        // if (_flow_id.flow_id == 0) { 
        //     printf("sim_time: %d, nxt_pkt_event: %d\n", _sim_time.sim_time, _nxt_pkt_event.nxt_pkt_event);
        // }
        // new arrival of ack packet
        uint16_t ack_pkt_cnt = 0;
        while (fetch_elem(_ack_buf, 0, ack_p)) {  // if ack pkt is available
            if ((ack_p.enqueue_time >= end_time) && (ack_p.enqueue_time > _nxt_pkt_event.nxt_pkt_event &&_snd_nxt.snd_nxt != _flow_size.flow_size)) {
                break;
            }

            // if (_flow_id.flow_id == 0) { 
            //     //printf("ack_p.enqueue_time: %ld, nxt_pkt_event: %ld\n", ack_p.enqueue_time, _nxt_pkt_event.nxt_pkt_event);
            // }

            _dequeue(_ack_buf, ack_p);

            if (ack_p.dst == _dst.dst) {// reaction for ACK/NACK packets
                #if PRINT_PKT_LOG
                PrintPkt(ack_p, "*ACK received by sender: ");
                #endif
                if (ack_p.pkt_type == PktType::ACK) { // ACK
                    _snd_una.snd_una = ack_p.sq_num;
                    printf("sender id: %d, variable set: snd_una: %ld\n", _src.src, _snd_una.snd_una);

                    if (_snd_una.snd_una >= _flow_size.flow_size) {
                        _flow_state = FlowState::COMPLETE;
                        // _nxt_pkt_event.nxt_pkt_event = INT64_MAX;
                        if (_flow_size.flow_size != 0) {
                            printf("sender id: %d, flow is finished\n", _src.src, _snd_una.snd_una);
                        }
                        break;
                    }

                    _last_ack_timestamp.last_ack_timestamp = _sim_time.sim_time;
                }
                else if (ack_p.pkt_type == PktType::NACK) { // NACK
                    _snd_nxt.snd_nxt = ack_p.sq_num;
                    _snd_una.snd_una = ack_p.sq_num;
                    //printf("variable set: snd_una: %ld, snd_nxt: %ld\n", _snd_una.snd_una,_snd_nxt.snd_nxt);

                    _last_ack_timestamp.last_ack_timestamp = _sim_time.sim_time;
                }
                else if (ack_p.pkt_type == PktType::CNP){ // CNP
                    _cc_para.CNPState = true;
                    dcqcn_rate_decrease(_cc_para, ack_p.src, _sim_time.sim_time);
                    dcqcn_alpha_update(_cc_para);
                    _cc_para.CNPState = false;
                    //record the rate is decreased
                    //use it to determine whether up
                    _cc_para.dcqcn_RateIsDecreased = true;
                }
                else { //PFC pause/resume packet 
                    if (ack_p.pkt_type == PktType::PFC_PAUSE) {
                        _pfc_state = PFCState::PAUSE;
                    }
                    else {
                         _pfc_state = PFCState::RESUME;
                    }
                }
            }
            else { // the ACK/NACK packet is not for me, send it out by the sender's receiver 
                _enqueue(_snd_buf, ack_p);
            }
            _sim_time.sim_time += 1; // increase 1 ns
            ack_pkt_cnt += 1;
        }

        // DCQCN timer function
        // the order should be: check if decrease, update alpha, increase send rate
        dcqcn_timer_reset_if_decrease(_cc_para, _sim_time.sim_time);
        dcqcn_timer_alpha_update(_cc_para, _sim_time.sim_time);
        dcqcn_timer_rate_increase(_cc_para, _src.src, _sim_time.sim_time);


        // if (_sender_id.sender_id == 0) {
        //     printf("snd_una: %ld, snd_nxt: %ld\n", _snd_una.snd_una,_snd_nxt.snd_nxt);
        // }

        if (_snd_una.snd_una >= _flow_size.flow_size) {
            _flow_state = FlowState::COMPLETE;
            break;
        }

        if (_sim_time.sim_time > _nxt_pkt_event.nxt_pkt_event) {
            _nxt_pkt_event.nxt_pkt_event = _sim_time.sim_time;
        }
        else {
            _sim_time.sim_time = _nxt_pkt_event.nxt_pkt_event;
        }
        // } // cannot handle when _sim_time.sim_time == nxt_pkt_event

        if ((_sim_time.sim_time - _last_ack_timestamp.last_ack_timestamp) >= RETRANSMIT_TIMER) {
           _snd_nxt.snd_nxt = _snd_una.snd_una;
            _last_ack_timestamp.last_ack_timestamp = _sim_time.sim_time;
        }

        // there must no new send event in this time frame
        if (ack_pkt_cnt == 0 && _snd_nxt.snd_nxt == _flow_size.flow_size) {break;}

        // FIXME TBD
        // the beggining of flow
        if (_snd_nxt.snd_nxt == 0) { 
            if (_start_time.start_time >= _sim_time.sim_time && _start_time.start_time <= end_time) {
                _sim_time.sim_time = _start_time.start_time;
            }
            else if (_start_time.start_time > end_time) {
                break;
            }
        }
        //

        if (_pfc_state == PFCState::PAUSE) {
            printf("In send phase: PFCState::PAUSE\n");
        }

        if (_flow_state == FlowState::UNCOMPLETE && _sim_time.sim_time >= _start_time.start_time && _pfc_state == PFCState::RESUME) { // the flow is unfinished
            // flow is not finished
            int64_t remain_data_len = (_flow_size.flow_size -_snd_nxt.snd_nxt);

            // if (remain_data_len <= 0 ) {continue;}

            int64_t payload_len = remain_data_len;
            if (remain_data_len > (MTU-header_len)) {   
                payload_len = MTU-header_len;
            }
            uint16_t pkt_size = payload_len+header_len;
            int64_t dequeue_time = _sim_time.sim_time + (pkt_size*8*1.0/_nic_rate.nic_rate)*(1000*1000*1000); // unit is ns
            int64_t enqueue_time = dequeue_time + _HS_link_delay.HS_link_delay;
            PktType pkt_type = PktType::DATA;
            
            int64_t sq_num = _snd_nxt.snd_nxt;

            Pkt pkt;
            create_pkt(pkt, _src.src, _dst.dst, _l4_port.l4_port, header_len, payload_len, header_len+payload_len, sq_num, flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type, path, 0, ECN_MARK::NO);


            #if PRINT_PKT_LOG
            PrintPkt(pkt, "data pkt sent by sender: ");
            #endif
            _enqueue(_snd_buf, pkt);

           _snd_nxt.snd_nxt += payload_len;

            // nxt_pkt_event must be >= dequeue_time, because cur_rate(m_rate) must be <= nic_rate;
            _nxt_pkt_event.nxt_pkt_event = _sim_time.sim_time + (pkt_size*8*1.0/_cc_para.m_rate)*(1000*1000*1000);

            _sim_time.sim_time = _sim_time.sim_time > dequeue_time ? _sim_time.sim_time : dequeue_time;
            //if (pkt_cnt > 20) break;

            sent_bytes += pkt_size;

            // FIXME TBD
            if (_snd_nxt.snd_nxt == _flow_size.flow_size) {
                _flow_state = FlowState::COMPLETE;
                _stop_time.stop_time = _sim_time.sim_time;
                if (_flow_size.flow_size > 0) {
                    printf("Flow is finished: sndserver id: %d, recvServerID id: %d, sender npu id: %d, recv npu id: %d, flow id : %d, _start_time: %ld ns, stop time : %ld ns\n", _snd_server_id.snd_server_id, _recv_server_id.recv_server_id, _src.src, _dst.dst, _flow_id.flow_id, _start_time.start_time, _stop_time.stop_time);
                }
                break;
            }
        }
    }

    // if (_sender_id.sender_id == 0 || _sender_id.sender_id == 1) {
    // if (_sender_id.sender_id == 0) {
        // printf("sender: %d, sim_time: %ld,, send rate is %.2f Gbps\n", _sender_id.sender_id, _sim_time.sim_time, sent_bytes*8*1.0/_sim_time_per_update.sim_time_per_update);
    // }
    _sim_time.sim_time = end_time;
    if (_sim_time.sim_time > _nxt_pkt_event.nxt_pkt_event) {
        _nxt_pkt_event.nxt_pkt_event = _sim_time.sim_time;
    }
}


// inline void nic_forward(Engine &ctx, MountedFlows &_mounted_flows,
//                         NIC_ID &_nic_id, NICRate &_nic_rate, SimTime &_sim_time, 
//                         SimTimePerUpdate &_sim_time_per_update, BidPktBuf &_bid_pkt_buf) {
//     if (_nic_id.nic_id == 0) {
//         printf("\n*******Enter into nic_forward*********\n");
//         printf("start _sim_time.sim_time: %ld\n", _sim_time.sim_time);
//         printf("end_time: %ld\n", _sim_time.sim_time+_sim_time_per_update.sim_time_per_update);
//     }

//     uint32_t flow_num = get_queue_len(_mounted_flows);
//     // uint32_t flow_num = _mounted_flows.flow.size();
//     for (uint32_t i = 0; i < flow_num; i++) {
//         PktBuf npu_snd_buf = ctx.get<PktBuf>(_mounted_flows.flow[i]);
//         uint32_t pkt_num = get_queue_len(npu_snd_buf);
//         Pkt pkt;
//         for (uint32_t k = 0; k < pkt_num; k++) {
//             _dequeue(npu_snd_buf, pkt);
//             _enqueue(_bid_pkt_buf.snd_buf, pkt);
//         }
//     }
//     // _sim_time.sim_time += _sim_time_per_update.sim_time_per_update;
// }

inline void nic_forward(Engine &ctx, NIC_ID &_nic_id, NICRate &_nic_rate, SimTime &_sim_time, 
                        SimTimePerUpdate &_sim_time_per_update, BidPktBuf &_bid_pkt_buf) {
    // if (_nic_id.nic_id == 0) {
    //     printf("\n*******Enter into nic_forward*********\n");
    //     printf("start _sim_time.sim_time: %ld\n", _sim_time.sim_time);
    //     printf("end_time: %ld\n", _sim_time.sim_time+_sim_time_per_update.sim_time_per_update);
    // }

    Map<uint16_t, Entity> tmp_snd_flow = ctx.data().snd_flows[_nic_id.nic_id];
    for (auto it = tmp_snd_flow.begin(); it != tmp_snd_flow.end(); ++it) {
        auto& entry = *it;
        Entity flow_entt = entry.value;

        PktBuf npu_snd_buf = ctx.get<PktBuf>(flow_entt);
        uint32_t pkt_num = get_queue_len(npu_snd_buf);
        Pkt pkt;
        for (uint32_t k = 0; k < pkt_num; k++) {
            //npu_snd_buf --> _bid_pkt_buf.snd_buf
            _dequeue(npu_snd_buf, pkt);
            _enqueue(_bid_pkt_buf.snd_buf, pkt);
        }
    }

}


inline void insertionSort(PktBuf &data) {
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
}


inline void nic_transmit(Engine &ctx, NIC_ID &_nic_id, 
                         NICRate &_nic_rate, HSLinkDelay &_HS_link_delay,
                         SimTime &_sim_time, SimTimePerUpdate &_sim_time_per_update, 
                         BidPktBuf &_bid_pkt_buf, TXHistory &_tx_history, 
                         NextHop &_next_hop, Seed &_seed) {
    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;

    // if (_nic_id.nic_id == 0) {
    //     printf("\n*******Enter into nic_forward*********\n");
    //     printf("start _sim_time.sim_time: %ld\n", _sim_time.sim_time);
    //     printf("end_time: %ld\n", end_time);
    // }

    insertionSort(_bid_pkt_buf.snd_buf);

    Entity in_port = ctx.data().inPorts[_next_hop.next_hop];

    Pkt pkt;
    while(_sim_time.sim_time < end_time) {
        if(!fetch_elem(_bid_pkt_buf.snd_buf, 0, pkt)) break;
        if(pkt.enqueue_time >= end_time) break;
        _dequeue(_bid_pkt_buf.snd_buf, pkt);

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

        if (loc > 0) {
            TXElem tmp_tx_elem;
            rm_queue_range(_tx_history, tmp_tx_elem, 0, loc); // FIXME, only handle when using 1 priority queue
        }

        // tail drop
        if ((pkt.payload_len+now_queue_len) > BUF_SIZE) {
            continue;
        }

        pkt.dequeue_time = (pkt.enqueue_time > _sim_time.sim_time ? pkt.enqueue_time : _sim_time.sim_time) + \
                            (pkt.ip_pkt_len*8*1.0/_nic_rate.nic_rate)*(1000*1000*1000);
        pkt.enqueue_time = pkt.dequeue_time + _HS_link_delay.HS_link_delay; //FIXME

        TXElem tx_elem = {pkt.dequeue_time, pkt.ip_pkt_len};
        _enqueue(_tx_history, tx_elem); 

        // ECN marking  
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

        _enqueue(ctx.get<PktBuf>(in_port), pkt);

        _sim_time.sim_time = pkt.dequeue_time;
    }

    _sim_time.sim_time = end_time;
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
    // if (_global_port_id.global_port_id == 0) {
    //     printf("\n\n*******Enter into set_forward_plan sys*********\n\n");
    // }
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
    // if (_global_port_id.global_port_id == 0) {
    //     printf("*******Enter into _forward sys*********\n");
    // }
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
    // if (_global_port_id.global_port_id == 0) {
    //     printf("*******Enter into remove_pkts sys*********\n");
    //     // printf("end_time: %lld\n", end_time);
    // }


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


inline void transmit(Engine &ctx, SchedTrajType sched_traj_type, 
                     PortType port_type, LocalPortID &_local_port_id, 
                     GlobalPortID &_global_port_id, SwitchID &_switch_id, 
                     NextHop &_next_hop, NextHopType next_hop_type, 
                     PktQueue &_pkt_queue, TXHistory &_tx_history, 
                     SSLinkDelay &_ss_link_delay, LinkRate &_link_rate, 
                     SimTime &_sim_time, SimTimePerUpdate &_sim_time_per_update, Seed &_seed)
{


    // if (_global_port_id.global_port_id == 0) {
    //     printf("*******Enter into transmit sys*********\n");
    // }

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

            //move packet from this egress port to next hop port / the dest host
            Entity next_hop_ett;
            if (next_hop_type == NextHopType::SWITCH) {
                next_hop_ett = ctx.data().inPorts[_next_hop.next_hop];
                _enqueue(ctx.get<PktBuf>(next_hop_ett), pkt);
            }
            else {
                next_hop_ett = ctx.data()._nics[_next_hop.next_hop];
                _enqueue(ctx.get<BidPktBuf>(next_hop_ett).recv_buf, pkt);
            }
            // if (_global_port_id.global_port_id == 4) continue;
            //
            // if (pkt.pkt_type == PktType::ACK || pkt.pkt_type == PktType::NACK) {
            //     printf("pkt_type: %d, switch_id: %d, global_port_id: %d, NextHopType: %d, next hop link: %d, src: %d --> dst: %d, payload_len: %d, sq_num: %lld, dequeue_time: %lld, enqueue_time: %lld\n", \
            //             pkt.pkt_type, _switch_id.switch_id, _global_port_id.global_port_id, next_hop_type, _next_hop.next_hop, pkt.src, pkt.dst, pkt.payload_len, pkt.sq_num, pkt.dequeue_time, pkt.enqueue_time);
            // }

            
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


inline void nic_receive(Engine &ctx, NIC_ID &_nic_id,
                        BidPktBuf &_bid_pkt_buf, NICRate &_nic_rate,
                        SimTime &_sim_time, SimTimePerUpdate &_sim_time_per_update) {
    
    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    
    Pkt pkt;
    while (_sim_time.sim_time < end_time) {
        if(!fetch_elem(_bid_pkt_buf.recv_buf, 0, pkt)) break;
        if(pkt.enqueue_time >= end_time) break;
        _dequeue(_bid_pkt_buf.recv_buf, pkt);

        // recv flow entity
        //???????????????????????
        // Entity recv_flow =  ctx.data()._recv_flows[pkt.flow_id];
        Entity recv_flow = ctx.data().recv_flows[pkt.src][pkt.flow_id];
        
        if (recv_flow == Entity::none()) return;

        PktBuf recv_buf = ctx.get<PktBuf>(recv_flow);
        _enqueue(recv_buf, pkt);
        // PrintPath(pkt.path, pkt.path_len);
        if(pkt.enqueue_time >= end_time) break;
    }
    //_sim_time.sim_time = end_time;
}

// receive system: 
// 1) generate ACK/NACK/CNP; 
// 2) or move them into the ack buffer of receiver's sender 
inline void flow_receive(Engine &ctx, FlowID &_flow_id, PktBuf &_recv_queue,
                         NICRate &_nic_rate, HSLinkDelay &_HS_link_delay, 
                         SimTime &_sim_time, SimTimePerUpdate &_sim_time_per_update,
                         RecvBytes &_recv_bytes, L4Port &_l4_port,
                         LastCNPTimestamp &_last_cnp_timestamp) {

    int64_t end_time = _sim_time.sim_time + _sim_time_per_update.sim_time_per_update;
    // if (_flow_id.flow_id == 0) {
    //     printf("*******Enter into receive system*********\n");
    //     printf("start _sim_time.sim_time: %ld\n", _sim_time.sim_time);
    //     printf("end_time: %ld\n", end_time);
    // }

    Pkt pkt;
    uint8_t flow_priority = 0;
    uint16_t header_len = 40;
    while (_sim_time.sim_time < end_time) {
        if(!fetch_elem(_recv_queue, 0, pkt)) break;
        if(pkt.enqueue_time >= end_time) break;
        _dequeue(_recv_queue, pkt);

        // PrintPath(pkt.path, pkt.path_len);
        if(pkt.enqueue_time >= end_time) break;
        
        // Entity snd_flow_entt = ctx.data()._snd_flows[pkt.flow_id];
        Entity snd_flow_entt = ctx.data().snd_flows[pkt.src][pkt.flow_id];

        if (pkt.pkt_type != PktType::DATA) { //ACK/NACK/CNP pkt
            
            _enqueue(ctx.get<AckPktBuf>(snd_flow_entt), pkt); //enqueue into the ack_pkt_buf

            // printf("flow_id: %d\n", _flow_id.flow_id);
            #if PRINT_PKT_LOG
            PrintPkt(pkt, "ACK/NACK received by sender's receiver: ");
            #endif
        }
        else { //DATA pkt
            uint16_t pkt_size = pkt.header_len + pkt.payload_len;
            _sim_time.sim_time += (pkt_size*8*1.0/_nic_rate.nic_rate)*(1000*1000*1000);

            // printf("flow_id: %d\n", _flow_id.flow_id);
            // printf("src: %d recv: %d, expected_recv_bytes: %ld\n", recv_flow_queue.recv_flow[0].src, recv_flow_queue.recv_flow[0].dst, _recv_bytes.recv_bytes);
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
            if (pkt.sq_num == _recv_bytes.recv_bytes) { // packet order is right: generate ACK
                _recv_bytes.recv_bytes += pkt.payload_len;
                sq_num = _recv_bytes.recv_bytes;
                PktType pkt_type = PktType::ACK;

                create_pkt(ack_pkt, ack_src, ack_dst, \
                        _l4_port.l4_port, header_len, 0, header_len, sq_num, \
                        pkt.flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type, path, 0, ECN_MARK::NO);
            }
            else { // packets is out of order, NACK
                sq_num = _recv_bytes.recv_bytes;
                PktType pkt_type = PktType::NACK;

                create_pkt(ack_pkt, ack_src, ack_dst, \
                           _l4_port.l4_port, header_len, 0, header_len, sq_num, \
                           pkt.flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type, path, 0, ECN_MARK::NO);     
            }
            
            // Entity direct_connect_in_port = ctx.data().inPorts[_next_hop.next_hop];
            _enqueue(ctx.get<PktBuf>(snd_flow_entt), ack_pkt);
            #if PRINT_PKT_LOG
            PrintPkt(ack_pkt, "ACK/NACK generated by receiver: ");
            #endif
            // printf("pkt.ecn: %d\n", pkt.ecn);
            // packet with ecn marking, generate a cnp packet
            if (pkt.ecn == ECN_MARK::YES && (_sim_time.sim_time - _last_cnp_timestamp.last_cnp_timestamp) >= CNP_DURATION) { 
            // if (pkt.ecn == ECN_MARK::YES) { 
                Pkt cnp_pkt;
                PktType pkt_type = PktType::CNP;
                create_pkt(cnp_pkt, ack_src, ack_dst, \
                           _l4_port.l4_port, header_len, 0, header_len, sq_num, \
                           pkt.flow_id, enqueue_time, dequeue_time, flow_priority, pkt_type, path, 0, ECN_MARK::YES);
                // Entity direct_connect_in_port = ctx.data().inPorts[_next_hop.next_hop];
                _enqueue(ctx.get<PktBuf>(snd_flow_entt), cnp_pkt);
                #if PRINT_PKT_LOG
                PrintPkt(cnp_pkt, "CNP generated by receiver: ");
                #endif
                _last_cnp_timestamp.last_cnp_timestamp = _sim_time.sim_time;
            }
        }
    }
    _sim_time.sim_time = end_time;
}


///////////////////////////////////////////////////////////////////
// support for astrasim
const int INTS_PER_EVENT = 7;  // 每个事件7个整数
int parseMadronaEvents(const MadronaEvents &madronaEvents, MadronaEvent parsedEvents[], int maxEvents) {
    int validEvents = 0;
    for (int i = 0; i < maxEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex + INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        MadronaEvent event = {
            madronaEvents.events[baseIndex],
            madronaEvents.events[baseIndex + 1],
            madronaEvents.events[baseIndex + 2],
            madronaEvents.events[baseIndex + 3],
            madronaEvents.events[baseIndex + 4],
            madronaEvents.events[baseIndex + 5],
            madronaEvents.events[baseIndex + 6]
        };
        // 如果解析出的事件全为0,则忽略
        if (event.type == 0 && event.eventId == 0 &&
            event.time == 0 && event.src == 0 &&
            event.dst == 0 && event.size == 0 && event.port == 0) {
            continue;
        }
        parsedEvents[validEvents++] = event;
    }
    return validEvents;  // 返回有效事件的数量
}
void updateMadronaEvents(MadronaEvents &madronaEvents, const MadronaEvent parsedEvents[], int numEvents) {
    // 清空 madronaEvents
    for (int i = 0; i < 1000; ++i) {
        madronaEvents.events[i] = 0;
    }

    for (int i = 0; i < numEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex +  INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        madronaEvents.events[baseIndex] = parsedEvents[i].type;
        madronaEvents.events[baseIndex + 1] = parsedEvents[i].eventId;
        madronaEvents.events[baseIndex + 2] = parsedEvents[i].time;
        madronaEvents.events[baseIndex + 3] = parsedEvents[i].src;
        madronaEvents.events[baseIndex + 4] = parsedEvents[i].dst;
        madronaEvents.events[baseIndex + 5] = parsedEvents[i].size;
        madronaEvents.events[baseIndex + 6] = parsedEvents[i].port;
    }
}  

int parseMadronaEvents(const MadronaEventsResult &madronaEvents, MadronaEvent parsedEvents[], int maxEvents) {
    int validEvents = 0;
    for (int i = 0; i < maxEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex + INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        MadronaEvent event = {
            madronaEvents.events[baseIndex],
            madronaEvents.events[baseIndex + 1],
            madronaEvents.events[baseIndex + 2],
            madronaEvents.events[baseIndex + 3],
            madronaEvents.events[baseIndex + 4],
            madronaEvents.events[baseIndex + 5],
            madronaEvents.events[baseIndex + 6]
        };
        // 如果解析出的事件全为0,则忽略
        if (event.type == 0 && event.eventId == 0 &&
            event.time == 0 && event.src == 0 &&
            event.dst == 0 && event.size == 0 && event.port == 0) {
            continue;
        }
        parsedEvents[validEvents++] = event;
    }
    return validEvents;  // 返回有效事件的数量
}
void updateMadronaEvents(MadronaEventsResult &madronaEvents, const MadronaEvent parsedEvents[], int numEvents) {
    // 清空 madronaEvents
    for (int i = 0; i < 1000; ++i) {
        madronaEvents.events[i] = 0;
    }

    for (int i = 0; i < numEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex +  INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        madronaEvents.events[baseIndex] = parsedEvents[i].type;
        madronaEvents.events[baseIndex + 1] = parsedEvents[i].eventId;
        madronaEvents.events[baseIndex + 2] = parsedEvents[i].time;
        madronaEvents.events[baseIndex + 3] = parsedEvents[i].src;
        madronaEvents.events[baseIndex + 4] = parsedEvents[i].dst;
        madronaEvents.events[baseIndex + 5] = parsedEvents[i].size;
        madronaEvents.events[baseIndex + 6] = parsedEvents[i].port;
    }
}

int parseMadronaEvents(const MadronaEventsQueue &madronaEvents, MadronaEvent parsedEvents[], int maxEvents) {
    int validEvents = 0;
    for (int i = 0; i < maxEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex + INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        MadronaEvent event = {
            madronaEvents.events[baseIndex],
            madronaEvents.events[baseIndex + 1],
            madronaEvents.events[baseIndex + 2],
            madronaEvents.events[baseIndex + 3],
            madronaEvents.events[baseIndex + 4],
            madronaEvents.events[baseIndex + 5],
            madronaEvents.events[baseIndex + 6]
        };
        // 如果解析出的事件全为0,则忽略
        if (event.type == 0 && event.eventId == 0 &&
            event.time == 0 && event.src == 0 &&
            event.dst == 0 && event.size == 0 && event.port == 0) {
            continue;
        }
        parsedEvents[validEvents++] = event;
    }
    return validEvents;  // 返回有效事件的数量
}
void updateMadronaEvents(MadronaEventsQueue &madronaEvents, const MadronaEvent parsedEvents[], int numEvents) {
    // 清空 madronaEvents
    for (int i = 0; i < 1000; ++i) {
        madronaEvents.events[i] = 0;
    }

    for (int i = 0; i < numEvents; ++i) {
        int baseIndex = i * INTS_PER_EVENT;
        if (baseIndex +  INTS_PER_EVENT-1 >= 1000) break;  // 确保不超出边界
        madronaEvents.events[baseIndex] = parsedEvents[i].type;
        madronaEvents.events[baseIndex + 1] = parsedEvents[i].eventId;
        madronaEvents.events[baseIndex + 2] = parsedEvents[i].time;
        madronaEvents.events[baseIndex + 3] = parsedEvents[i].src;
        madronaEvents.events[baseIndex + 4] = parsedEvents[i].dst;
        madronaEvents.events[baseIndex + 5] = parsedEvents[i].size;
        madronaEvents.events[baseIndex + 6] = parsedEvents[i].port;
    }
}

void Comm_SetFlow(Engine &ctx,MadronaEvent event)
{
    Entity npu_entt = ctx.data()._npus[event.src];
    FlowEvent flowEvent;
    flowEvent.src = event.src;
    flowEvent.dst = event.dst;
    flowEvent.flow_size = event.size;
    flowEvent.l4_port = event.port;
    flowEvent.extra_1=event.eventId;
    // flowEvent.start_time = event.time + ctx.get<SimTime>(npu_entt).sim_time;
    flowEvent.start_time = event.time;
    _enqueue_flow(ctx.get<NewFlowQueue>(npu_entt),flowEvent);
}


inline void tick(Engine &ctx,
                 Action &action,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step,
                 Results &results,
                 Results2 &results2,
                 // SimulationTime
                 SimulationTime &time,
                 // internal event queue to save event excuted in future 
                 MadronaEventsQueue &madronaEventsQueue,
                 // external event to put in madronaEventsQueue
                 MadronaEvents &madronaEvents,
                 // event result at current time
                 MadronaEventsResult &madronaEventsResult,
                 ProcessParams &processParams
                 )
{


    time.time=ctx.get<SimTime>(ctx.data().inPorts[0]).sim_time;
    // printf("simulation_time: %ld\n",ctx.get<SimTime>(ctx.data()._npus[0]).sim_time);
    printf("gpu:\n");
    printf("schedule run: %d\n", processParams.params[0]);
    printf("flow run: %d\n", processParams.params[1]);
    printf("simulation_time: %ld\n",time.time);

    

// test
// for (int i = 0; i < 6; i++)
// {
//     printf("%d",madronaEvents.events[i]);
// }

    // printf("parse madronaEvents\n");
    const int maxEvents = 1000 / 7;
    MadronaEvent parsedEvents[maxEvents];
    // parse MadronaEvents
    int validEvents = parseMadronaEvents(madronaEvents, parsedEvents, maxEvents);
    if (validEvents == 0)
    {
        // printf("MadronaEvents is empty!\n ");
    }
    // add madronaEvents to madronaEventsQueue
    else
    {
        printf("receiveEvents : %d\n",validEvents);
        // print events
        for (int i = 0; i < validEvents; ++i)
        {
            printf("Event %d: type=%d, eventId=%d, time=%d, src=%d, dst=%d, size=%d, port=%d\n",
                   i, parsedEvents[i].type, parsedEvents[i].eventId,
                   parsedEvents[i].time, parsedEvents[i].src,
                   parsedEvents[i].dst, parsedEvents[i].size,parsedEvents[i].port);
        }
        MadronaEvent exitedEvents[maxEvents];
        // add receive events to event queue.
        int existedEventsNum = parseMadronaEvents(madronaEventsQueue, exitedEvents, maxEvents);
        // printf("eventsQueueNum : %d\n",existedEventsNum);
        for (int i = 0; i < validEvents; ++i)
        {
            // relative time changes to absolute time
            parsedEvents[i].time+=time.time;
            exitedEvents[existedEventsNum+i]=parsedEvents[i];
        }
        updateMadronaEvents(madronaEventsQueue,exitedEvents,existedEventsNum+validEvents);
        // clear recieve events.
        updateMadronaEvents(madronaEvents, parsedEvents, 0);
        // // for instance: change one event property
        // if (validEvents > 0)
        // {
        //     parsedEvents[0].time += 1;
        // }
        // updateMadronaEvents(madronaEvents, parsedEvents, validEvents);
    }

    // 1 frame 1 ns
    // time.time = time.time + 1;

    MadronaEvent eventsQueue[maxEvents];
    // process event queue
    int eventsQueueNum = parseMadronaEvents(madronaEventsQueue, eventsQueue, maxEvents);
    printf("eventsQueueNum :%d\n",eventsQueueNum);
    MadronaEvent eventsResult[maxEvents];
    int resultIndex = 0;
    MadronaEvent eventsFuture[maxEvents];
    int futureIndex = 0;
    for (int i = 0; i < eventsQueueNum; i++)
    {
        // printf("time.time:%ld\n",time.time);
        // printf("eventsQueue[%d].time:%d,type:%d,id:%d\n",i,eventsQueue[i].time,eventsQueue[i].type,eventsQueue[i].eventId);
        // to do :maybe relative time,done
        if (time.time >= eventsQueue[i].time)
        {
            // sim_schedule event.
            if (eventsQueue[i].type == 2)
            {
                eventsResult[resultIndex] = eventsQueue[i];
                resultIndex++;
                printf("process sim_schedule event. time:%ld, eventId=%d.\n,", eventsResult[resultIndex].time,eventsResult[resultIndex].eventId);
            }
            // sim_send event.
            else if (eventsQueue[i].type == 0)
            {
                // to do : 1 set flow entities
                Comm_SetFlow(ctx,eventsQueue[i]);
                printf("process sim_send event,time:%ld, eventId=%d.\n", time.time,eventsQueue[i].eventId);
                // // assume flow finish at 20 frame after.
                // if (time.time >= eventsQueue[i].time + 20)
                // {
                //     // to do : 1 set flow entities
                //     Comm_SetFlow(ctx,eventsQueue[i]);
                //     // to do : 2 set result to eventsResult when flow done
                //     eventsResult[resultIndex] = eventsQueue[i];
                //     eventsResult[resultIndex].time = eventsQueue[i].time + 20;
                //     resultIndex++;
                //     printf("process sim_send event,time:%ld.\n", time.time);
                // }
                // else
                // {
                //     eventsFuture[futureIndex] = eventsQueue[i];
                //     futureIndex++;
                // }
            }
        }
        else
        {
            eventsFuture[futureIndex] = eventsQueue[i];
            futureIndex++;
        }
    }

    printf("check flow done\n");

    //support for astrasim: check flow done
    uint32_t num = ctx.data().num_npu;
    for (uint32_t i = 0; i < num; i++)
    {
        Entity npu_entt = ctx.data()._npus[i];
        CompletedFlowQueue completedFlowQueue = ctx.get<CompletedFlowQueue>(npu_entt);

        uint32_t flow_event_num = get_queue_len(completedFlowQueue);
        // printf("completedFlowQueue.flow_event_num=%d.\n", flow_event_num);
        if (flow_event_num > 0)
        {
            for (uint32_t i = 0; i < flow_event_num; i++)
            {
                FlowEvent flow_event;
                _dequeue_flow(completedFlowQueue, flow_event);
                MadronaEvent event_temp;
                event_temp.type = 0;
                event_temp.eventId = flow_event.extra_1;
                event_temp.time = time.time;
                eventsResult[resultIndex] = event_temp;
                resultIndex++;
                printf("finish sim_send event,time:%ld.\n", time.time);
            }

            clear_queue(completedFlowQueue);
        }
    }

    printf("check flow done over\n");

    // put to madronaEventsResult
    if (resultIndex > 0)
    {
        printf("finishFlowEvents : %d\n",resultIndex);
        // print events
        for (int i = 0; i < resultIndex; ++i)
        {
            printf("Event %d: type=%d, eventId=%d, time=%d, src=%d, dst=%d, size=%d, port=%d\n",
                   i, eventsResult[i].type, eventsResult[i].eventId,
                   eventsResult[i].time, eventsResult[i].src,
                   eventsResult[i].dst, eventsResult[i].size,eventsResult[i].port);
        }
        // printf("resultNum:%d\n", resultIndex + 1);
        updateMadronaEvents(madronaEventsResult, eventsResult, resultIndex + 1);
    }

    // printf("futureNum:%d\n", futureIndex + 1);
    updateMadronaEvents(madronaEventsQueue, eventsFuture, futureIndex + 1);
}





void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
{
    printf("*******Enter into setupTasks*********\n");



    // auto receive_sys = builder.addToGraph<ParallelForNode<Engine, receive, ReceiverID, NextHop, PktBuf, \
    //                                        NICRate, HSLinkDelay, SimTime, SimTimePerUpdate, RecvFlowQueue>>({receive_sys}); 

    // auto send_sys = builder.addToGraph<ParallelForNode<Engine, send, SenderID, NextHop, NICRate, \
    //                                    HSLinkDelay, SimTime, SimTimePerUpdate, SendFlowQueue, PktBuf>>({receive_sys});

    auto comm_sys=builder.addToGraph<ParallelForNode<Engine, tick, Action, Reset, GridPos, Reward, Done, CurStep,Results,Results2,SimulationTime,MadronaEventsQueue,MadronaEvents,MadronaEventsResult,ProcessParams>>({});

    auto setup_flow_sys = builder.addToGraph<ParallelForNode<Engine, setup_flow, NPU_ID, NewFlowQueue, \
                                             SimTime, SimTimePerUpdate>>({comm_sys}); 

    // auto nic_receive_sys = builder.addToGraph<ParallelForNode<Engine, nic_receive, NIC_ID, \
    //                                           BidPktBuf, NICRate, SimTime, SimTimePerUpdate>>({setup_flow_sys});

    // auto flow_receive_sys = builder.addToGraph<ParallelForNode<Engine, flow_receive, FlowID, PktBuf, \
    //                                        NICRate, HSLinkDelay, SimTime, SimTimePerUpdate, \
    //                                        RecvBytes, L4Port, LastCNPTimestamp>>({nic_receive_sys});

    auto flow_send_sys = builder.addToGraph<ParallelForNode<Engine, flow_send, FlowID, Src, Dst, // dst npu
                                                            L4Port, // layer 4 port
                                                            FlowSize, StartTime, StopTime,
                                                            NIC_ID, SndServerID, RecvServerID, // host pair id
                                                            SndNxt, SndUna, FlowState,
                                                            LastAckTimestamp, NxtPktEvent, PFCState, CC_Para,
                                                            PktBuf, AckPktBuf,
                                                            NICRate, HSLinkDelay, SimTime, SimTimePerUpdate>>({setup_flow_sys});

    auto check_flow_state_sys = builder.addToGraph<ParallelForNode<Engine, check_flow_state, NPU_ID, CompletedFlowQueue, \
                                                   SimTime, SimTimePerUpdate>>({flow_send_sys}); 

    auto nic_forward_sys = builder.addToGraph<ParallelForNode<Engine, nic_forward, \
                                              NIC_ID, NICRate, SimTime, \
                                              SimTimePerUpdate, BidPktBuf>>({check_flow_state_sys});

    auto nic_transmit_sys = builder.addToGraph<ParallelForNode<Engine, nic_transmit, NIC_ID, \
                                               NICRate, HSLinkDelay, \
                                               SimTime, SimTimePerUpdate, \
                                               BidPktBuf, TXHistory, \
                                               NextHop, Seed>>({nic_forward_sys});

    auto set_forward_plan_sys = builder.addToGraph<ParallelForNode<Engine, set_forward_plan, \
                                                   LocalPortID, GlobalPortID, \
                                                   SwitchID, PktBuf, ForwardPlan, \
                                                   SimTime, SimTimePerUpdate>>({nic_transmit_sys});

    auto forward_sys = builder.addToGraph<ParallelForNode<Engine, _forward, SchedTrajType, \
                                          LocalPortID, GlobalPortID, PktQueue, SwitchID, \
                                          SimTime, SimTimePerUpdate>>({set_forward_plan_sys});

    auto remove_pkts_sys = builder.addToGraph<ParallelForNode<Engine, remove_pkts, \
                                              LocalPortID, GlobalPortID, PktBuf, ForwardPlan, \
                                              SimTime, SimTimePerUpdate>>({forward_sys}); 

    auto transmit_sys = builder.addToGraph<ParallelForNode<Engine, transmit, SchedTrajType, \
                                           PortType, LocalPortID,  GlobalPortID, SwitchID, \
                                           NextHop, NextHopType, PktQueue, TXHistory, \
                                           SSLinkDelay, LinkRate, SimTime, \
                                           SimTimePerUpdate, Seed>>({remove_pkts_sys}); 
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
    ctx.get<Results>(agent).results = 0.f;
    ctx.get<SimulationTime>(agent).time = 0;
    //***************************************************
    // for the router forwarding function, of GPU_acclerated DES
    numInPort = 0;
    numEPort = 0;

    // numSender = 0;
    // numReceiver = 0;
    numSwitch = 0;  

    // num_snd_flow = 0;
    // num_recv_flow = 0;

    num_nic = 0;
    num_npu = 0;

    printf("Gen entity as following: \n");
    generate_switch(ctx, K_ARY);
    generate_in_port(ctx, K_ARY);
    generate_e_port(ctx, K_ARY);
    generate_host(ctx, K_ARY);

    printf("Gen configuration is as following: \n");
    // printf("Total %d switches, %d in_ports, %d e_ports, %d hosts", \
    //        ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, \
    //        ctx.data().num_snd_flow, ctx.data().num_recv_flow, ctx.data().num_nic, \
    //        ctx.data().num_npu);
    printf("Total %d switches, %d in_ports, %d e_ports, %d nics, %d npus\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, \
           ctx.data().num_nic, ctx.data().num_npu);

    //generate_fib_table(ctx, K_ARY);
    //***************************************************

}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
