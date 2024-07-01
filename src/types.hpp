#pragma once

#include <madrona/components.hpp>

#define SEND_RATE 1000LL*1000*1000*100 // 100 Gbps

#define PKT_BUF_LEN 1000 // 1024*1
#define TX_BUF_LEN 1000

#define INPORT_NUM  16 //40

#define SS_LINK_DELAY 1000 //ns switch --> switch
#define LOOKAHEADT_TIME SS_LINK_DELAY
#define HS_LINK_DELAY 1000 //ns host --> switch
#define PACKET_PROCESS_TIME 100 //100 ns
#define TMP_BUF_LEN SS_LINK_DELAY/PACKET_PROCESS_TIME


#define K_ARY 4 //8
// number of dst host
#define NUM_HOST K_ARY*K_ARY*K_ARY/4
#define MAX_NUM_NEXT_HOP K_ARY
//


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

enum class Action : int32_t {
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
   int32_t sender_id;
};

struct EgressPortID {
   int32_t egress_port_id;
};


enum class NextHopType : uint32_t {
    HOST = 0,
    SWITCH = 1,
};

struct NextHop {
    int32_t next_hop;
};

struct HSLinkDelay {
    int64_t HS_link_delay;
};

struct NICRate {
    int64_t nic_rate;
};


struct Pkt {
    int32_t src_id;
    int32_t dest_id;
    int16_t pkt_size;
    int16_t flow_id;
    int64_t enqueue_time;
    int64_t dequeue_time;
};
struct PktBuf {
    Pkt pkts[PKT_BUF_LEN];
    uint16_t head;
    uint16_t tail;
    Pkt tmp_pkt_buf[TMP_BUF_LEN];
    uint16_t t_pkt_buf_len;
    // uint32_t d_positions[INPORT_NUM];
    // uint32_t d_positions_end[INPORT_NUM];
};

struct SimTime {
    int64_t sim_time;
};
struct SimTimePerUpdate {
    int64_t sim_time_per_update;
};


// for sender system, of GPU_acclerated DES
struct Sender : public madrona::Archetype<
    SenderID, //
    // EgressPortID,// directed-connected 
    NextHopType,
    NextHop,
    SimTime,
    SimTimePerUpdate,
    HSLinkDelay, // link delay between host and switch 
    NICRate
> {};


struct ReceiverID {
   int32_t receiver_id;
};


struct IngressPortID {
   int32_t ingress_port_id;
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
    NICRate
> {};



enum class PortType : uint32_t {
    InPort = 0,
    EPort = 1,
};

// for the router forwarding system, of GPU_acclerated DES
struct PortID {
   int32_t port_id;
};


struct TXElem {
    int64_t dequeue_time;
    int16_t pkt_size;
};
struct TXHistory {
    TXElem pkts[TX_BUF_LEN];
    uint16_t head;
    uint16_t tail;
};

enum class SchedTrajType : uint32_t {
    FIFO = 0,
    DRR = 1,
};


enum class SwitchType : uint32_t {
    Edge = 0,
    Aggr = 1,
    Core = 2,
};

struct SwitchID {
    int32_t switch_id;
};


struct ForwardPlan {
    uint16_t forward_plan[INPORT_NUM];
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

struct Switch : public madrona::Archetype<
    SwitchType, //
    SwitchID, //
    FIBTable
> {};


struct IngressPort : public madrona::Archetype<
    PortType,
    PortID,
    SwitchID, //
    PktBuf, //
    ForwardPlan, //
    SimTime,
    SimTimePerUpdate
> {};


struct EgressPort : public madrona::Archetype<
    SchedTrajType,
    PortType,
    PortID,
    SwitchID, //
    PktBuf, //
    TXHistory,
    NextHopType,
    NextHop,
    LinkRate,
    SSLinkDelay,
    SimTime,
    SimTimePerUpdate
> {};

}



//***************************************************