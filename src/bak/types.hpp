#pragma once

#include <madrona/components.hpp>


#define K_ARY 4 // 8 //4 //16
#define INPORT_NUM  K_ARY //40

#define NIC_RATE 1000LL*1000*1000 * 100 // 100 Gbps
#define BW 100 // 100 Gbps
// egress port
#define QUEUE_NUM 1


// packet buffer
#define PKT_BUF_LEN 200 // 1024*1
#define TX_BUF_LEN PKT_BUF_LEN

#define BUF_SIZE PKT_BUF_LEN*1000  // 200*1460 //byte
#define PFC_THR 80

// ECN configuration
#define K_MAX 1000 * 1024   // 400 * 4 * 1024
#define K_MIN 200 * 1024   // 100 * 4 * 1024
#define P_MAX 0.2
 
// link
#define SS_LINK_DELAY 1000 // 1000 //ns switch --> switch
#define HS_LINK_DELAY 1000  // 1000  //ns host --> switch
#define MTU 1500

#define LOOKAHEAD_TIME SS_LINK_DELAY

#define PACKET_PROCESS_TIME 100 //100 ns
#define TMP_BUF_LEN SS_LINK_DELAY/PACKET_PROCESS_TIME

// sender
#define FLOW_NUM 1 //how many flows concurrent exist in 1 sender at most 


// number of dst host
#define NUM_HOST K_ARY*K_ARY*K_ARY/4
#define MAX_NUM_NEXT_HOP K_ARY
//

#define MAX_PATH_LEN 20 //16 //8 hops (each hop includes device id and port id)

//Retransmit timer
#define RETRANSMIT_TIMER 100*1000 // 100us



// DCQCN parameter
#define DCQCN_Alpha 1.0
#define DCQCN_CurrentRate  NIC_RATE // 100Gbps 
#define DCQCN_TargetRate  NIC_RATE // 100Gbps
#define DCQCN_G 0.00390625
#define DCQCN_AIRate 20*1000*1000 // 20M bps  //50M bps
#define DCQCN_HAIRate 200*1000*1000 // 200 Mbps  //500M bps
#define DCQCN_AlphaUpdateInterval 1 * 1000 // 1us,  nanoseconds
#define DCQCN_RateDecreaseInterval 4 * 1000 // 4us,  nanoseconds
#define DCQCN_RateIncreaseInterval 300 * 1000 // 300us,  nanoseconds

#define DCQCN_IncreaseStageCount 0 // runtime variables, should set at runtime
#define DCQCN_RecoveryStageThreshhold 1 // 5

#define DCQCN_RateDecreaseNextTime 0 // runtime variables, should set at runtime
#define DCQCN_RateIncreaseNextTime 0 // runtime variables, should set at runtime
#define DCQCN_AlphaUpdateNextTime 0 // runtime variables, should set at runtime
#define DCQCN_RateIsDecreased false

#define CNP_DURATION 50*1000 // 50*1000 //50us


namespace madsimple {

enum class ExportID : uint32_t {
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    NumExports,
};

struct Reset {
    int32_t resetNow;
};

enum class Action : uint32_t {
    Up    = 0,
    Down  = 1,
    Left  = 2,
    Right = 3,
    None,
};

struct GridPos {
    int32_t y;
    int32_t x;
};

struct Reward {
    float r;
};

struct Done {
    float episodeDone;
};

struct CurStep {
    uint32_t step;
};

struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep
> {};



//********************************************************
struct SenderID {
   uint32_t sender_id;
};

struct EgressPortID {
   uint32_t egress_port_id;
};


enum class NextHopType : uint8_t {
    HOST = 0,
    SWITCH = 1,
};

struct NextHop {
    uint32_t next_hop;
};

struct HSLinkDelay {
    int64_t HS_link_delay;
};

struct NICRate {
    int64_t nic_rate;
};

enum class PktType : uint8_t {
    DATA = 0,
    ACK = 1,
    PFC_PAUSE = 2,
    PFC_RESUME = 3,
    NACK = 4,
    CNP = 5,
};

enum class ECN_MARK: uint8_t {
    NO = 0,
    YES = 1,
};

struct Pkt {
    uint32_t src;   
    uint32_t dst;  // also PFC_FOR_UPSTREAM for the PFC frame 
    uint16_t l4_port;
    uint16_t header_len;
    uint16_t payload_len;
    uint16_t ip_pkt_len;
    int64_t sq_num;
    uint16_t flow_id;  // also the idx of priority queue for the PFC frame 
    int64_t enqueue_time;
    int64_t dequeue_time;
    uint8_t flow_priority;
    PktType pkt_type;  // also PFC state for the PFC frame 

    uint32_t path[MAX_PATH_LEN];
    uint8_t path_len;

    ECN_MARK ecn;
};
struct PktBuf {
    Pkt pkts[PKT_BUF_LEN];
    uint16_t head;
    uint16_t tail;
    uint16_t cur_num;
    uint32_t cur_bytes;
    // Pkt tmp_pkt_buf[TMP_BUF_LEN];
    // uint16_t t_pkt_buf_len;
    // uint32_t d_positions[INPORT_NUM];
    // uint32_t d_positions_end[INPORT_NUM];
};

enum class PFCState : uint8_t {
    PAUSE = 2,
    RESUME = 3,
};
struct PktQueue {
    PktBuf pkt_buf[QUEUE_NUM];
    PFCState queue_pfc_state[QUEUE_NUM];
};


struct SimTime {
    int64_t sim_time;
};
struct SimTimePerUpdate {
    int64_t sim_time_per_update;
};

enum class FlowState : uint8_t {
    UNCOMPLETE = 0,
    COMPLETE = 1,
};

struct SendFlow {
    uint32_t src;
    uint32_t dst;
    uint16_t l4_port;
    uint16_t nic; // nic id
    int64_t flow_size;
    int64_t sq_num;
    int64_t snd_nxt;
    int64_t snd_una;
    FlowState flow_state;
    PFCState pfc_state;

    int64_t last_ack_timestamp; // retransmit timer
    int64_t nxt_pkt_event; // the timestamp of the sending event for the next packet

    int64_t m_rate; // current rate/Mpbs
    int64_t tar_rate; // target rate/Mbps

    double dcqcn_Alpha; // the alpha parameter in DCQCN
    double dcqcn_G; // the G parameter in DCQCN
    int64_t dcqcn_AIRate; // the increase rate in additive increase stage
    int64_t dcqcn_HAIRate; // the increase rate in hyper additive increase stage
    uint32_t dcqcn_AlphaUpdateInterval; // the timer interval to update alpha
    uint32_t dcqcn_RateDecreaseInterval; // the timer interval to decrease target rate
    uint32_t dcqcn_RateIncreaseInterval; // the timer interval to increase target rate
    uint32_t dcqcn_IncreaseStageCount; // record the count increase stage
    uint32_t dcqcn_RecoveryStageThreshold; // the threshshold of reconvery stage
    int64_t dcqcn_AlphaUpdateNextTime; // the next time (nanoseconds) to update alpha parameter
    int64_t dcqcn_RateDecreaseNextTime; // the next time (nanoseconds) to decrease target rate
    int64_t dcqcn_RateIncreaseNextTime; // the next time (nanoseconds) to increase target rate
    bool dcqcn_RateIsDecreased; // the flag to indicate rate is decreased when received ECE flag
    bool CNPState; 
};

// flow state
struct SendFlowQueue {
    SendFlow send_flow[FLOW_NUM];
    uint16_t head;
    uint16_t tail;
};

struct RecvFlow {
    uint32_t src;
    uint32_t dst;
    uint16_t l4_port;
    uint16_t nic; // nic id
    int64_t recv_bytes;
    //int64_t sq_num;
    //int64_t snd_nxt;
    //int64_t snnd_una;
    FlowState flow_state;
    int64_t last_cnp_timestamp;
};

// flow state
struct RecvFlowQueue {
    RecvFlow recv_flow[FLOW_NUM];
    uint16_t head;
    uint16_t tail;
};

// for sender system, of GPU_acclerated DES
struct Sender : public madrona::Archetype<
    SenderID, //
    NextHopType,
    NextHop,
    SimTime,
    SimTimePerUpdate,
    HSLinkDelay, // link delay between host and switch 
    NICRate,
    SendFlowQueue,
    PktBuf
> {};


struct ReceiverID {
   uint32_t receiver_id;
};


struct IngressPortID {
   uint32_t ingress_port_id;
};


struct Receiver : public madrona::Archetype<
    ReceiverID, //
    IngressPortID, // directed-connected 
    PktBuf,
    NextHopType,
    NextHop,
    SimTime,
    SimTimePerUpdate,
    HSLinkDelay,
    NICRate,
    RecvFlowQueue
> {};


enum class PortType : uint8_t {
    InPort = 0,
    EPort = 1,
};

// for the router forwarding system, of GPU_acclerated DES
struct LocalPortID {
   uint32_t local_port_id;
};
struct GlobalPortID {
   uint32_t global_port_id;
};

struct TXElem {
    int64_t dequeue_time;
    uint16_t ip_pkt_len;
};
struct TXHistory {
    TXElem pkts[TX_BUF_LEN];
    uint16_t head;
    uint16_t tail;
    uint16_t cur_num;
    uint32_t cur_bytes;
};

enum class SchedTrajType : uint8_t {
    SP = 0,
    FIFO = 1,
};


enum class SwitchType : uint8_t {
    Edge = 0,
    Aggr = 1,
    Core = 2,
};

struct SwitchID {
    uint32_t switch_id;
};


struct ForwardPlan {
    uint64_t forward_plan[INPORT_NUM];
};


struct FIBTable {
    uint8_t fib_table[NUM_HOST][MAX_NUM_NEXT_HOP];
};


struct LinkRate {
    int64_t link_rate;
};

struct SSLinkDelay {
    int64_t SS_link_delay;
};

struct QueueNumPerPort {
    uint8_t queue_num_per_port;
};

struct Seed {
    uint32_t seed;
};

struct QLen {
    uint32_t q_len;
};

struct Switch : public madrona::Archetype<
    SwitchType, //
    SwitchID, //
    FIBTable,
    QueueNumPerPort
> {};


struct IngressPort : public madrona::Archetype<
    PortType,
    LocalPortID,
    GlobalPortID,
    SwitchID, //
    PktBuf, //
    ForwardPlan, //
    SimTime,
    SimTimePerUpdate
> {};


struct EgressPort : public madrona::Archetype<
    SchedTrajType,
    PortType,
    LocalPortID,
    GlobalPortID,
    SwitchID, 
    PktQueue,
    TXHistory,
    QLen,
    NextHopType,
    NextHop,
    LinkRate,
    SSLinkDelay,
    SimTime,
    SimTimePerUpdate,
    Seed
> {};

}



//***************************************************