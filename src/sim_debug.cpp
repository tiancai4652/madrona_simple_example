#include "sim_debug.hpp"

namespace madsimple {

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
                                double, double) {}

}
