#pragma once

#include "../sim.hpp"
#include "sys_flow.hpp"

namespace madsimple::net_sys_interface {

uint64_t getCurrentTime(Engine &ctx);

bool isExistedFlow(Engine &ctx);

void addSimtime(Engine &ctx, uint64_t delta_ns);

void setFlow(Engine &ctx,
             uint32_t npu_id,
             uint64_t comm_src,
             uint64_t comm_dst,
             uint64_t flow_size,
             uint32_t flow_id,
             uint64_t comm_para);

uint32_t checkFlowFinish(Engine &ctx,
                         uint32_t npu_id,
                         SysFlow flows_finish[]);

uint32_t checkFlowFinish(Engine &ctx,
                         uint32_t npu_id,
                         SysFlow flows_finish[],
                         uint32_t max_flows_finish);

void clearFlowFinishQueue(Engine &ctx, uint32_t npu_id);

bool addEvent(Engine &ctx, uint64_t event_time_ns);

uint64_t msToNs(Time time_ms);

Time nsToMs(uint64_t time_ns);

}
