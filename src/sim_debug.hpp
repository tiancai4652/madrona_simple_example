#pragma once

#include <cstdint>

#include "sim.hpp"

namespace madsimple {

extern const bool init_log_print_enabled;
extern const bool system_log_print_enabled;

bool systemLogEnabled(const char *scope, uint64_t step);

void printInitTopoLog(const Sim &sim, Engine &ctx);
void printInitFlowLog(const Sim &sim);

void printSystemBegin(uint64_t step, Time now, const char *scope, const char *phase);
void printSystemEnd(uint64_t step, Time now, const char *scope, const char *phase);
void printSystemScheduleFlow(uint64_t step, Time now, const FlowDef &flow);
void printSystemScheduleSummary(uint64_t step, Time now,
                                int32_t scheduled_count,
                                int32_t pending_before,
                                int32_t pending_after,
                                int32_t delayed_before,
                                int32_t delayed_after,
                                int32_t flow_routes_before,
                                int32_t flow_routes_after);
void printSystemDeliverArrival(uint64_t step, Time now, const FlowArrivalEv &ev);
void printSystemDeliverBwUpdate(uint64_t step, Time now, const BwUpdateEv &ev);
void printSystemDeliverPfc(uint64_t step, Time now, const PfcControlEv &ev);
void printSystemDeliverSummary(uint64_t step, Time now,
                               int32_t delayed_before,
                               int32_t delayed_after,
                               int32_t inbox_arrival_count,
                               int32_t inbox_bwupdate_count,
                               int32_t inbox_pfc_count);
void printSystemArrivalTag(uint64_t step, Time now, const char *action,
                           const FlowTagState &tag, int32_t dirty);
void printSystemArrivalSummary(uint64_t step, Time now,
                               int32_t created_count,
                               int32_t updated_count,
                               int32_t skipped_count);
void printSystemBwUpdateTag(uint64_t step, Time now, const char *action,
                            const FlowTagState &tag, int32_t dirty);
void printSystemBwUpdateForward(uint64_t step, Time now,
                                FlowId flow_id,
                                int32_t port_id,
                                int32_t next_port_id);
void printSystemBwUpdateComplete(uint64_t step, Time now, FlowId flow_id);
void printSystemBwUpdateSummary(uint64_t step, Time now,
                                int32_t created_count,
                                int32_t updated_count,
                                int32_t buffered_zero_count,
                                int32_t destroyed_count,
                                int32_t forwarded_count,
                                int32_t completed_count,
                                int32_t skipped_count);
void printSystemPfcState(uint64_t step, Time now, const PfcControlEv &ev,
                         int32_t applied_paused, int32_t dirty);
void printSystemPfcSummary(uint64_t step, Time now,
                           int32_t applied_count,
                           int32_t skipped_count);
void printSystemAllocPort(uint64_t step, Time now,
                          int32_t port_id,
                          double port_bw,
                          int32_t num_tags,
                          int32_t num_live,
                          double sum_in,
                          double sum_out,
                          int32_t qos_mode,
                          int32_t is_dest_only,
                          int32_t has_buffer);
void printSystemAllocSummary(uint64_t step, Time now,
                             int32_t dirty_port_count,
                             int32_t processed_port_count,
                             int32_t dirty_tag_count);
void printSystemEmitSummary(uint64_t step, Time now,
                            int32_t dirty_port_count,
                            int32_t arrival_emit_count,
                            int32_t bwupdate_emit_count);
void printSystemPfcDetectSummary(uint64_t step, Time now,
                                 int32_t checked_port_count,
                                 int32_t emitted_pfc_count,
                                 int32_t pause_timer_count,
                                 int32_t resume_timer_count);
void printSystemClearSummary(uint64_t step, Time now,
                             int32_t cleared_port_count);
void printSystemDTSummary(uint64_t step, Time now,
                          double delayed_gap,
                          double pending_gap,
                          double finish_gap,
                          double drain_gap,
                          double backlog_gap,
                          double pfc_pause_gap,
                          double pfc_resume_gap,
                          double chosen_dt);
void printSystemBufferSummary(uint64_t step, Time now,
                              int32_t processed_port_count,
                              int32_t destroy_count,
                              double total_buf_cnt);
void printSystemProgressSummary(uint64_t step, Time now,
                                double dt,
                                int32_t finished_source_count,
                                int32_t emitted_cleanup_count,
                                double next_now,
                                double next_finish_gap);

}
