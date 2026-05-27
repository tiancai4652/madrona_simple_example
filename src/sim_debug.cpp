#include "sim_debug.hpp"

#if !defined(__CUDA_ARCH__)
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#endif

namespace madsimple {

namespace {

#if !defined(__CUDA_ARCH__)
inline bool envFlagEnabled(const char *name)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return false;
    }

    return !(env[0] == '0' && env[1] == '\0');
}

inline bool scopeMatches(const char *want, const char *actual)
{
    return std::strcmp(want, actual) == 0;
}
#endif

}

#if defined(__CUDA_ARCH__)
const bool init_log_print_enabled = false;
const bool system_log_print_enabled = false;

bool systemLogEnabled(const char *, uint64_t)
{
    return false;
}

void printInitTopoLog(const Sim &, Engine &) {}
void printInitFlowLog(const Sim &) {}
void printSystemBegin(uint64_t, Time, const char *, const char *) {}
void printSystemEnd(uint64_t, Time, const char *, const char *) {}
void printSystemScheduleFlow(uint64_t, Time, const FlowDef &) {}
void printSystemScheduleSummary(uint64_t, Time, int32_t, int32_t, int32_t,
                                int32_t, int32_t, int32_t, int32_t) {}
void printSystemDeliverArrival(uint64_t, Time, const FlowArrivalEv &) {}
void printSystemDeliverBwUpdate(uint64_t, Time, const BwUpdateEv &) {}
void printSystemDeliverPfc(uint64_t, Time, const PfcControlEv &) {}
void printSystemDeliverSummary(uint64_t, Time, int32_t, int32_t, int32_t,
                               int32_t, int32_t) {}
void printSystemArrivalTag(uint64_t, Time, const char *,
                           const FlowTagState &, int32_t) {}
void printSystemArrivalSummary(uint64_t, Time, int32_t, int32_t, int32_t) {}
void printSystemBwUpdateTag(uint64_t, Time, const char *,
                            const FlowTagState &, int32_t) {}
void printSystemBwUpdateForward(uint64_t, Time, FlowId, int32_t, int32_t) {}
void printSystemBwUpdateComplete(uint64_t, Time, FlowId) {}
void printSystemBwUpdateSummary(uint64_t, Time, int32_t, int32_t, int32_t,
                                int32_t, int32_t, int32_t, int32_t) {}
void printSystemPfcState(uint64_t, Time, const PfcControlEv &, int32_t,
                         int32_t) {}
void printSystemPfcSummary(uint64_t, Time, int32_t, int32_t) {}
void printSystemAllocPort(uint64_t, Time, int32_t, double, int32_t, int32_t,
                          double, double, int32_t, int32_t, int32_t) {}
void printSystemAllocSummary(uint64_t, Time, int32_t, int32_t, int32_t) {}
void printSystemEmitSummary(uint64_t, Time, int32_t, int32_t, int32_t) {}
void printSystemEmitBwUpdateTag(uint64_t, Time, int32_t, FlowId, double,
                                double) {}
void printSystemPfcDetectSummary(uint64_t, Time, int32_t, int32_t, int32_t,
                                 int32_t) {}
void printSystemClearSummary(uint64_t, Time, int32_t) {}
void printSystemDTSummary(uint64_t, Time, double, double, double, double,
                          double, double, double, double) {}
void printSystemBufferSummary(uint64_t, Time, int32_t, int32_t, double) {}
void printSystemProgressSummary(uint64_t, Time, double, int32_t, int32_t,
                                int32_t, int32_t, int32_t, double, double) {}
#else
const bool init_log_print_enabled = envFlagEnabled("init_log_print_enabled");
const bool system_log_print_enabled = envFlagEnabled("system_log_print_enabled");

bool systemLogEnabled(const char *scope, uint64_t step)
{
    if (!system_log_print_enabled) {
        return false;
    }

    const char *scope_env = std::getenv("system_log_scope");
    if (scope_env != nullptr && scope_env[0] != '\0' &&
        !scopeMatches(scope_env, "all") && !scopeMatches(scope_env, scope)) {
        return false;
    }

    const char *step_env = std::getenv("system_log_step");
    if (step_env != nullptr && step_env[0] != '\0') {
        char *end = nullptr;
        unsigned long want = std::strtoul(step_env, &end, 10);
        if (end != step_env && *end == '\0' && step != want) {
            return false;
        }
    }

    const char *every_env = std::getenv("system_log_every");
    if (every_env != nullptr && every_env[0] != '\0') {
        char *end = nullptr;
        unsigned long every = std::strtoul(every_env, &end, 10);
        if (end != every_env && *end == '\0' && every > 0 &&
            step % every != 0) {
            return false;
        }
    }

    return true;
}

void printInitTopoLog(const Sim &, Engine &) {}
void printInitFlowLog(const Sim &) {}

void printSystemBegin(uint64_t step, Time now, const char *scope,
                      const char *phase)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BEGIN] step=" << step
              << " now=" << now
              << " scope=" << scope
              << " phase=" << phase
              << "\n";
}

void printSystemEnd(uint64_t step, Time now, const char *scope,
                    const char *phase)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][END] step=" << step
              << " now=" << now
              << " scope=" << scope
              << " phase=" << phase
              << "\n";
}

void printSystemScheduleFlow(uint64_t, Time, const FlowDef &) {}
void printSystemDeliverArrival(uint64_t, Time, const FlowArrivalEv &) {}
void printSystemDeliverBwUpdate(uint64_t, Time, const BwUpdateEv &) {}
void printSystemDeliverPfc(uint64_t, Time, const PfcControlEv &) {}
void printSystemArrivalTag(uint64_t, Time, const char *,
                           const FlowTagState &, int32_t) {}
void printSystemBwUpdateTag(uint64_t, Time, const char *,
                            const FlowTagState &, int32_t) {}
void printSystemBwUpdateForward(uint64_t, Time, FlowId, int32_t, int32_t) {}
void printSystemBwUpdateComplete(uint64_t, Time, FlowId) {}
void printSystemPfcState(uint64_t, Time, const PfcControlEv &, int32_t,
                         int32_t) {}
void printSystemAllocPort(uint64_t, Time, int32_t, double, int32_t, int32_t,
                          double, double, int32_t, int32_t, int32_t) {}
void printSystemEmitBwUpdateTag(uint64_t, Time, int32_t, FlowId, double,
                                double) {}

void printSystemScheduleSummary(uint64_t step, Time now,
                                int32_t scheduled_count,
                                int32_t pending_before,
                                int32_t pending_after,
                                int32_t delayed_before,
                                int32_t delayed_after,
                                int32_t flow_routes_before,
                                int32_t flow_routes_after)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][SCHEDULE][SUMMARY] step=" << step
              << " now=" << now
              << " scheduled_count=" << scheduled_count
              << " pending_before=" << pending_before
              << " pending_after=" << pending_after
              << " delayed_before=" << delayed_before
              << " delayed_after=" << delayed_after
              << " flow_routes_before=" << flow_routes_before
              << " flow_routes_after=" << flow_routes_after
              << "\n";
}

void printSystemDeliverSummary(uint64_t step, Time now,
                               int32_t delayed_before,
                               int32_t delayed_after,
                               int32_t inbox_arrival_count,
                               int32_t inbox_bwupdate_count,
                               int32_t inbox_pfc_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][DELIVER][SUMMARY] step=" << step
              << " now=" << now
              << " delayed_before=" << delayed_before
              << " delayed_after=" << delayed_after
              << " inbox_arrival_count=" << inbox_arrival_count
              << " inbox_bwupdate_count=" << inbox_bwupdate_count
              << " inbox_pfc_count=" << inbox_pfc_count
              << "\n";
}

void printSystemArrivalSummary(uint64_t step, Time now,
                               int32_t created_count,
                               int32_t updated_count,
                               int32_t skipped_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][ARRIVAL][SUMMARY] step=" << step
              << " now=" << now
              << " created_count=" << created_count
              << " updated_count=" << updated_count
              << " skipped_count=" << skipped_count
              << "\n";
}

void printSystemBwUpdateSummary(uint64_t step, Time now,
                                int32_t created_count,
                                int32_t updated_count,
                                int32_t buffered_zero_count,
                                int32_t destroyed_count,
                                int32_t forwarded_count,
                                int32_t completed_count,
                                int32_t skipped_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BWUPDATE][SUMMARY] step=" << step
              << " now=" << now
              << " created_count=" << created_count
              << " updated_count=" << updated_count
              << " buffered_zero_count=" << buffered_zero_count
              << " destroyed_count=" << destroyed_count
              << " forwarded_count=" << forwarded_count
              << " completed_count=" << completed_count
              << " skipped_count=" << skipped_count
              << "\n";
}

void printSystemPfcSummary(uint64_t step, Time now,
                           int32_t applied_count,
                           int32_t skipped_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][PFC][SUMMARY] step=" << step
              << " now=" << now
              << " applied_count=" << applied_count
              << " skipped_count=" << skipped_count
              << "\n";
}

void printSystemAllocSummary(uint64_t step, Time now,
                             int32_t dirty_port_count,
                             int32_t processed_port_count,
                             int32_t dirty_tag_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][ALLOC][SUMMARY] step=" << step
              << " now=" << now
              << " dirty_port_count=" << dirty_port_count
              << " processed_port_count=" << processed_port_count
              << " dirty_tag_count=" << dirty_tag_count
              << "\n";
}

void printSystemEmitSummary(uint64_t step, Time now,
                            int32_t dirty_port_count,
                            int32_t arrival_emit_count,
                            int32_t bwupdate_emit_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][EMIT][SUMMARY] step=" << step
              << " now=" << now
              << " dirty_port_count=" << dirty_port_count
              << " arrival_emit_count=" << arrival_emit_count
              << " bwupdate_emit_count=" << bwupdate_emit_count
              << "\n";
}

void printSystemPfcDetectSummary(uint64_t step, Time now,
                                 int32_t checked_port_count,
                                 int32_t emitted_pfc_count,
                                 int32_t pause_timer_count,
                                 int32_t resume_timer_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][PFCDETECT][SUMMARY] step=" << step
              << " now=" << now
              << " checked_port_count=" << checked_port_count
              << " emitted_pfc_count=" << emitted_pfc_count
              << " pause_timer_count=" << pause_timer_count
              << " resume_timer_count=" << resume_timer_count
              << "\n";
}

void printSystemClearSummary(uint64_t step, Time now,
                             int32_t cleared_port_count)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][CLEAR][SUMMARY] step=" << step
              << " now=" << now
              << " cleared_port_count=" << cleared_port_count
              << "\n";
}

void printSystemDTSummary(uint64_t step, Time now,
                          double delayed_gap,
                          double pending_gap,
                          double finish_gap,
                          double drain_gap,
                          double backlog_gap,
                          double pfc_pause_gap,
                          double pfc_resume_gap,
                          double chosen_dt)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][DT][SUMMARY] step=" << step
              << " now=" << now
              << " delayed_gap=" << delayed_gap
              << " pending_gap=" << pending_gap
              << " finish_gap=" << finish_gap
              << " drain_gap=" << drain_gap
              << " backlog_gap=" << backlog_gap
              << " pfc_pause_gap=" << pfc_pause_gap
              << " pfc_resume_gap=" << pfc_resume_gap
              << " chosen_dt=" << chosen_dt
              << "\n";
}

void printSystemBufferSummary(uint64_t step, Time now,
                              int32_t processed_port_count,
                              int32_t destroy_count,
                              double total_buf_cnt)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][BUFFER][SUMMARY] step=" << step
              << " now=" << now
              << " processed_port_count=" << processed_port_count
              << " destroy_count=" << destroy_count
              << " total_buf_cnt=" << total_buf_cnt
              << "\n";
}

void printSystemProgressSummary(uint64_t step, Time now,
                                double dt,
                                int32_t finished_source_count,
                                int32_t emitted_cleanup_count,
                                int32_t source_scan_count,
                                int32_t source_destroy_count,
                                int32_t buffered_dirty_port_count,
                                double next_now,
                                double next_finish_gap)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[SYS][PROGRESS][SUMMARY] step=" << step
              << " now=" << now
              << " dt=" << dt
              << " source_scan_count=" << source_scan_count
              << " finished_source_count=" << finished_source_count
              << " source_destroy_count=" << source_destroy_count
              << " emitted_cleanup_count=" << emitted_cleanup_count
              << " buffered_dirty_port_count=" << buffered_dirty_port_count
              << " next_now=" << next_now
              << " next_finish_gap=" << next_finish_gap
              << "\n";
}
#endif

}
