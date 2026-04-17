#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

int32_t Sim::findBacklogDrainTimerIndex(int32_t port_id) const
{
    for (int32_t i = 0; i < numBacklogDrainTimers; i++) {
        if (backlogDrainPortIDs[i] == port_id) {
            return i;
        }
    }
    return -1;
}

int32_t Sim::findPfcPauseTimerIndex(int32_t ingress_port_id) const
{
    for (int32_t i = 0; i < numPfcPauseTimers; i++) {
        if (pfcPausePortIDs[i] == ingress_port_id) {
            return i;
        }
    }
    return -1;
}

int32_t Sim::findPfcResumeTimerIndex(int32_t ingress_port_id) const
{
    for (int32_t i = 0; i < numPfcResumeTimers; i++) {
        if (pfcResumePortIDs[i] == ingress_port_id) {
            return i;
        }
    }
    return -1;
}

void Sim::setBacklogDrainTimer(int32_t port_id, Time t)
{
    int32_t idx = findBacklogDrainTimerIndex(port_id);
    if (idx >= 0) {
        backlogDrainTimers[idx] = t;
        return;
    }
    if (numBacklogDrainTimers < MAX_TOPO_PORTS) {
        backlogDrainPortIDs[numBacklogDrainTimers] = port_id;
        backlogDrainTimers[numBacklogDrainTimers] = t;
        numBacklogDrainTimers += 1;
    }
}

void Sim::setPfcPauseTimer(int32_t ingress_port_id, Time t)
{
    int32_t idx = findPfcPauseTimerIndex(ingress_port_id);
    if (idx >= 0) {
        pfcPauseTimers[idx] = t;
        return;
    }
    if (numPfcPauseTimers < MAX_TOPO_PORTS) {
        pfcPausePortIDs[numPfcPauseTimers] = ingress_port_id;
        pfcPauseTimers[numPfcPauseTimers] = t;
        numPfcPauseTimers += 1;
    }
}

void Sim::setPfcResumeTimer(int32_t ingress_port_id, Time t)
{
    int32_t idx = findPfcResumeTimerIndex(ingress_port_id);
    if (idx >= 0) {
        pfcResumeTimers[idx] = t;
        return;
    }
    if (numPfcResumeTimers < MAX_TOPO_PORTS) {
        pfcResumePortIDs[numPfcResumeTimers] = ingress_port_id;
        pfcResumeTimers[numPfcResumeTimers] = t;
        numPfcResumeTimers += 1;
    }
}

void Sim::clearBacklogDrainTimer(int32_t port_id)
{
    int32_t idx = findBacklogDrainTimerIndex(port_id);
    if (idx < 0) {
        return;
    }
    for (int32_t i = idx + 1; i < numBacklogDrainTimers; i++) {
        backlogDrainPortIDs[i - 1] = backlogDrainPortIDs[i];
        backlogDrainTimers[i - 1] = backlogDrainTimers[i];
    }
    numBacklogDrainTimers -= 1;
}

void Sim::clearPfcPauseTimer(int32_t ingress_port_id)
{
    int32_t idx = findPfcPauseTimerIndex(ingress_port_id);
    if (idx < 0) {
        return;
    }
    for (int32_t i = idx + 1; i < numPfcPauseTimers; i++) {
        pfcPausePortIDs[i - 1] = pfcPausePortIDs[i];
        pfcPauseTimers[i - 1] = pfcPauseTimers[i];
    }
    numPfcPauseTimers -= 1;
}

void Sim::clearPfcResumeTimer(int32_t ingress_port_id)
{
    int32_t idx = findPfcResumeTimerIndex(ingress_port_id);
    if (idx < 0) {
        return;
    }
    for (int32_t i = idx + 1; i < numPfcResumeTimers; i++) {
        pfcResumePortIDs[i - 1] = pfcResumePortIDs[i];
        pfcResumeTimers[i - 1] = pfcResumeTimers[i];
    }
    numPfcResumeTimers -= 1;
}

void Sim::pfcPropagateSystem(Context &ctx)
{
    constexpr const char *scope = "ingress_chain";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    int32_t applied_count = 0;
    int32_t skipped_count = 0;

    if (log_enabled) {
        printSystemBegin(step, now, scope, "pfc_propagate");
    }

    if (enablePfc == 0 || numInboxPfc == 0) {
        if (log_enabled) {
            printSystemPfcSummary(step, now, applied_count, skipped_count);
            printSystemEnd(step, now, scope, "pfc_propagate");
        }
        return;
    }

    for (int32_t i = 0; i < numInboxPfc; i++) {
        const PfcControlEv &ev = inboxPfc[i];
        if (ev.target_port_id < 0 || ev.target_port_id >= numPorts) {
            skipped_count += 1;
            continue;
        }

        Entity target_port = portEntities[ev.target_port_id];
        if (target_port == Entity::none()) {
            skipped_count += 1;
            continue;
        }

        PortPfcState &state = ctx.get<PortPfcState>(target_port);
        if (ev.priority >= 0 && ev.priority < PFC_MAX_PRIORITY) {
            state.paused[ev.priority] = ev.paused;
            ctx.get<DirtyPort>(target_port).isDirty = 1;
            applied_count += 1;
            if (log_enabled) {
                printSystemPfcState(step, now, ev,
                    state.paused[ev.priority],
                    ctx.get<DirtyPort>(target_port).isDirty);
            }
        } else {
            skipped_count += 1;
        }
    }

    if (log_enabled) {
        printSystemPfcSummary(step, now, applied_count, skipped_count);
        printSystemEnd(step, now, scope, "pfc_propagate");
    }
}

void Sim::pfcThresholdDetectSystem(Context &ctx)
{
    constexpr const char *scope = "emit_pfc";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);
    int32_t checked_port_count = 0;
    int32_t emitted_pfc_count = 0;

    if (enablePfc == 0) {
        if (log_enabled) {
            printSystemPfcDetectSummary(step, now, 0, 0,
                numPfcPauseTimers, numPfcResumeTimers);
        }
        return;
    }

    if (pfcEgress != 0) {
        for (int32_t port_id = 0; port_id < numPorts; port_id++) {
            Entity egress_e = portEntities[port_id];
            if (egress_e == Entity::none() || ctx.get<DirtyPort>(egress_e).isDirty == 0) {
                continue;
            }

            PortPfcConfig &cfg = ctx.get<PortPfcConfig>(egress_e);
            if (cfg.pfc_enabled == 0) {
                continue;
            }
            checked_port_count += 1;
            PortPfcState &state = ctx.get<PortPfcState>(egress_e);

            double buf_by_pri[PFC_MAX_PRIORITY] {};
            int32_t upstream_ports[MAX_TOPO_PORTS] {};
            int32_t num_upstream_ports = 0;

            for (int32_t i = 0; i < numTagIndexEntries; i++) {
                if (tagIndex[i].port_id != port_id) {
                    continue;
                }
                Entity te = tagIndex[i].entity;
                if (te == Entity::none()) {
                    continue;
                }
                FlowTagState &tag = ctx.get<FlowTagState>(te);
                if (tag.is_source != 0) {
                    continue;
                }
                materializeBacklog(tag, now);
                int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                buf_by_pri[pri] += tag.backlog;

                int32_t ip = tag.ingress_port_id;
                if (ip >= 0 && ip < numPorts) {
                    int32_t up = peerPort[ip];
                    if (up >= 0) {
                        bool dup = false;
                        for (int32_t j = 0; j < num_upstream_ports; j++) {
                            if (upstream_ports[j] == up) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup && num_upstream_ports < MAX_TOPO_PORTS) {
                            upstream_ports[num_upstream_ports++] = up;
                        }
                    }
                }
            }

            bool state_changed = false;
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                double buf = buf_by_pri[pri];
                if (state.pause_active[pri] == 0 && buf >= cfg.xoff[pri] - 0.5) {
                    state.pause_active[pri] = 1;
                    state_changed = true;
                    state.paused_upstream_count[pri] = 0;
                    for (int32_t j = 0; j < num_upstream_ports && j < MAX_PAUSED_UPSTREAMS; j++) {
                        int32_t up = upstream_ports[j];
                        state.paused_upstreams[pri][state.paused_upstream_count[pri]++] = up;
                        int32_t detect_slot = findNodeSlot(portToNode[port_id]);
                        int32_t upstream_slot = findNodeSlot(portToNode[up]);
                        Time pfc_delay = defaultLinkDelay;
                        if (detect_slot >= 0 && upstream_slot >= 0 && linkDelays[detect_slot][upstream_slot] >= 0.0) {
                            pfc_delay = linkDelays[detect_slot][upstream_slot];
                        }
                        DelayedEvent ev {};
                        ev.t = computePropagationTime(pfc_delay);
                        ev.type = DelayedEvent::Type::PfcControl;
                        ev.pfcctrl = PfcControlEv {
                            .target_port_id = up,
                            .source_port_id = port_id,
                            .priority = pri,
                            .paused = 1,
                        };
                        pushDelayedEvent(ev);
                        emitted_pfc_count += 1;
                    }
                    state.pfc_cnt[pri] += 1;
                } else if (state.pause_active[pri] != 0 && buf <= cfg.xon[pri] + 0.5) {
                    state.pause_active[pri] = 0;
                    state_changed = true;
                    for (int32_t j = 0; j < state.paused_upstream_count[pri]; j++) {
                        int32_t up = state.paused_upstreams[pri][j];
                        int32_t detect_slot = findNodeSlot(portToNode[port_id]);
                        int32_t upstream_slot = findNodeSlot(portToNode[up]);
                        Time pfc_delay = defaultLinkDelay;
                        if (detect_slot >= 0 && upstream_slot >= 0 && linkDelays[detect_slot][upstream_slot] >= 0.0) {
                            pfc_delay = linkDelays[detect_slot][upstream_slot];
                        }
                        DelayedEvent ev {};
                        ev.t = computePropagationTime(pfc_delay);
                        ev.type = DelayedEvent::Type::PfcControl;
                        ev.pfcctrl = PfcControlEv {
                            .target_port_id = up,
                            .source_port_id = port_id,
                            .priority = pri,
                            .paused = 0,
                        };
                        pushDelayedEvent(ev);
                        emitted_pfc_count += 1;
                    }
                    state.paused_upstream_count[pri] = 0;
                }
            }

            if (state_changed) {
                clearPfcPauseTimer(port_id);
                clearPfcResumeTimer(port_id);
            }
        }
        return;
    }

    int32_t ingress_check[MAX_TOPO_PORTS] {};
    int32_t num_ingress_check = 0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none() || ctx.get<DirtyPort>(port_e).isDirty == 0) {
            continue;
        }
        for (int32_t i = 0; i < numTagIndexEntries; i++) {
            if (tagIndex[i].port_id != port_id) {
                continue;
            }
            FlowTagState &tag = ctx.get<FlowTagState>(tagIndex[i].entity);
            int32_t ip = tag.ingress_port_id;
            if (ip >= 0) {
                bool dup = false;
                for (int32_t j = 0; j < num_ingress_check; j++) {
                    if (ingress_check[j] == ip) {
                        dup = true;
                        break;
                    }
                }
                if (!dup && num_ingress_check < MAX_TOPO_PORTS) {
                    ingress_check[num_ingress_check++] = ip;
                }
            }
        }
    }

    for (int32_t i = 0; i < num_ingress_check; i++) {
        int32_t ingress_port = ingress_check[i];
        if (ingress_port < 0 || ingress_port >= numPorts) {
            continue;
        }
        Entity ingress_e = portEntities[ingress_port];
        if (ingress_e == Entity::none()) {
            continue;
        }

        PortPfcConfig &cfg = ctx.get<PortPfcConfig>(ingress_e);
        if (cfg.pfc_enabled == 0) {
            continue;
        }
        checked_port_count += 1;
        PortPfcState &state = ctx.get<PortPfcState>(ingress_e);

        double buf_by_pri[PFC_MAX_PRIORITY] {};
        double net_rate_by_pri[PFC_MAX_PRIORITY] {};
        for (int32_t j = 0; j < numIngressTags; j++) {
            if (ingressTags[j].ingress_port_id != ingress_port) {
                continue;
            }
            Entity te = ingressTags[j].entity;
            if (te == Entity::none()) {
                continue;
            }
            FlowTagState &tag = ctx.get<FlowTagState>(te);
            if (tag.is_source != 0) {
                continue;
            }
            materializeBacklog(tag, now);
            int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
            buf_by_pri[pri] += tag.backlog;
            net_rate_by_pri[pri] += (tag.in_bw - tag.out_bw);
        }

        int32_t upstream_port = -1;
        if (ingress_port >= 0 && ingress_port < numPorts) {
            upstream_port = peerPort[ingress_port];
        }

        bool state_changed = false;
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            double buf = buf_by_pri[pri];
            if (state.pause_active[pri] == 0 && buf >= cfg.xoff[pri] - 0.5) {
                state.pause_active[pri] = 1;
                state_changed = true;
                state.paused_upstream_count[pri] = 0;
                if (upstream_port >= 0) {
                    state.paused_upstreams[pri][state.paused_upstream_count[pri]++] = upstream_port;
                    int32_t detect_slot = findNodeSlot(portToNode[ingress_port]);
                    int32_t upstream_slot = findNodeSlot(portToNode[upstream_port]);
                    Time pfc_delay = defaultLinkDelay;
                    if (detect_slot >= 0 && upstream_slot >= 0 && linkDelays[detect_slot][upstream_slot] >= 0.0) {
                        pfc_delay = linkDelays[detect_slot][upstream_slot];
                    }
                    DelayedEvent ev {};
                    ev.t = computePropagationTime(pfc_delay);
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = upstream_port,
                        .source_port_id = ingress_port,
                        .priority = pri,
                        .paused = 1,
                    };
                    pushDelayedEvent(ev);
                    emitted_pfc_count += 1;
                }
                state.pfc_cnt[pri] += 1;
            } else if (state.pause_active[pri] != 0 && buf <= cfg.xon[pri] + 0.5) {
                state.pause_active[pri] = 0;
                state_changed = true;
                for (int32_t k = 0; k < state.paused_upstream_count[pri]; k++) {
                    int32_t up = state.paused_upstreams[pri][k];
                    int32_t detect_slot = findNodeSlot(portToNode[ingress_port]);
                    int32_t upstream_slot = findNodeSlot(portToNode[up]);
                    Time pfc_delay = defaultLinkDelay;
                    if (detect_slot >= 0 && upstream_slot >= 0 && linkDelays[detect_slot][upstream_slot] >= 0.0) {
                        pfc_delay = linkDelays[detect_slot][upstream_slot];
                    }
                    DelayedEvent ev {};
                    ev.t = computePropagationTime(pfc_delay);
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = up,
                        .source_port_id = ingress_port,
                        .priority = pri,
                        .paused = 0,
                    };
                    pushDelayedEvent(ev);
                    emitted_pfc_count += 1;
                }
                state.paused_upstream_count[pri] = 0;
            }
        }

        if (state_changed) {
            clearPfcPauseTimer(ingress_port);
            clearPfcResumeTimer(ingress_port);
        }

        // Rebuild timers from the post-transition state so a newly entered
        // pause period can immediately schedule its matching resume timer.
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            double buf = buf_by_pri[pri];
            double net_rate = net_rate_by_pri[pri];
            if (state.pause_active[pri] == 0 && net_rate > 1.0) {
                double gap = cfg.xoff[pri] - buf;
                if (gap > 1e-9) {
                    double t_xoff = gap / net_rate;
                    if (t_xoff > 1e-9 && t_xoff < 1e6) {
                        setPfcPauseTimer(ingress_port, t_xoff);
                    }
                }
            } else if (state.pause_active[pri] != 0) {
                double effective_net = net_rate;
                if (effective_net >= -1e-15) {
                    double out_total = 0.0;
                    for (int32_t j = 0; j < numIngressTags; j++) {
                        if (ingressTags[j].ingress_port_id != ingress_port) {
                            continue;
                        }
                        Entity te = ingressTags[j].entity;
                        if (te != Entity::none()) {
                            FlowTagState &t = ctx.get<FlowTagState>(te);
                            if (t.priority == pri) {
                                out_total += t.out_bw;
                            }
                        }
                    }
                    if (out_total > 1e-15) {
                        effective_net = -out_total;
                    }
                }
                if (effective_net < -1.0) {
                    double gap = buf - cfg.xon[pri];
                    if (gap > 1e-9) {
                        double t_xon = gap / (-effective_net);
                        if (t_xon > 1e-9 && t_xon < 1e6) {
                            setPfcResumeTimer(ingress_port, t_xon);
                        }
                    }
                }
            }
        }
    }

    if (log_enabled) {
        printSystemPfcDetectSummary(step, now, checked_port_count,
            emitted_pfc_count, numPfcPauseTimers, numPfcResumeTimers);
    }
}

}
