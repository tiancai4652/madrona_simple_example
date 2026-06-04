#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

MADRONA_NO_INLINE int32_t Sim::countActiveBacklogDrainTimers(
    Context &ctx) const
{
    int32_t count = 0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e != Entity::none() &&
            timerIsActive(ctx.get<PortTimers>(port_e).backlog_drain)) {
            count += 1;
        }
    }
    return count;
}

MADRONA_NO_INLINE int32_t Sim::countActivePfcPauseTimers(Context &ctx) const
{
    int32_t count = 0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e != Entity::none() &&
            timerIsActive(ctx.get<PortTimers>(port_e).pfc_pause)) {
            count += 1;
        }
    }
    return count;
}

MADRONA_NO_INLINE int32_t Sim::countActivePfcResumeTimers(Context &ctx) const
{
    int32_t count = 0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e != Entity::none() &&
            timerIsActive(ctx.get<PortTimers>(port_e).pfc_resume)) {
            count += 1;
        }
    }
    return count;
}

MADRONA_NO_INLINE bool Sim::hasActiveBacklogDrainTimers(Context &ctx) const
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e != Entity::none() &&
            timerIsActive(ctx.get<PortTimers>(port_e).backlog_drain)) {
            return true;
        }
    }
    return false;
}

MADRONA_NO_INLINE bool Sim::hasActivePfcPauseTimers(Context &ctx) const
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e != Entity::none() &&
            timerIsActive(ctx.get<PortTimers>(port_e).pfc_pause)) {
            return true;
        }
    }
    return false;
}

MADRONA_NO_INLINE bool Sim::hasActivePfcResumeTimers(Context &ctx) const
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e != Entity::none() &&
            timerIsActive(ctx.get<PortTimers>(port_e).pfc_resume)) {
            return true;
        }
    }
    return false;
}

MADRONA_NO_INLINE void Sim::setBacklogDrainTimer(PortTimers &timers, Time t)
{
    if (!timerIsActive(timers.backlog_drain) ||
        t < timers.backlog_drain) {
        timers.backlog_drain = t;
    }
}

MADRONA_NO_INLINE void Sim::setPfcPauseTimer(PortTimers &timers, Time t)
{
    timers.pfc_pause = t;
}

MADRONA_NO_INLINE void Sim::setPfcResumeTimer(PortTimers &timers, Time t)
{
    timers.pfc_resume = t;
}

MADRONA_NO_INLINE void Sim::clearBacklogDrainTimer(PortTimers &timers)
{
    timers.backlog_drain = timerInactiveSentinel();
}

MADRONA_NO_INLINE void Sim::clearPfcPauseTimer(PortTimers &timers)
{
    timers.pfc_pause = timerInactiveSentinel();
}

MADRONA_NO_INLINE void Sim::clearPfcResumeTimer(PortTimers &timers)
{
    timers.pfc_resume = timerInactiveSentinel();
}

MADRONA_NO_INLINE void Sim::applyDrainHintOnePort(
    int32_t port_id,
    PortDrainHint &hint,
    PortTimers &timers)
{
    (void)port_id;
    if (hint.want_clear != 0) {
        clearBacklogDrainTimer(timers);
    }
    if (hint.want_set != 0) {
        setBacklogDrainTimer(timers, hint.set_t);
    }

    hint.want_clear = 0;
    hint.want_set = 0;
    hint.set_t = 0.0;
}

MADRONA_NO_INLINE void Sim::applyPfcTimerOnePort(
    int32_t ingress_port_id,
    PortPfcState &state,
    PortTimers &timers)
{
    (void)ingress_port_id;
    if (state.want_clear_pause != 0) {
        clearPfcPauseTimer(timers);
    }
    if (state.want_clear_resume != 0) {
        clearPfcResumeTimer(timers);
    }
    if (state.want_set_pause != 0) {
        setPfcPauseTimer(timers, state.set_pause_t);
    }
    if (state.want_set_resume != 0) {
        setPfcResumeTimer(timers, state.set_resume_t);
    }

    state.want_clear_pause = 0;
    state.want_clear_resume = 0;
    state.want_set_pause = 0;
    state.want_set_resume = 0;
    state.set_pause_t = 0.0;
    state.set_resume_t = 0.0;
}

// Phase C: per-Port PFC threshold detect worker. Only writes this port's
// own PortPfcState / PortOutbox / PortTraceLast. Reads DirtyPort of other
// ports as read-only (pfcPropagate earlier is the only writer in-frame
// and finished before this node; clearDirtyOnePort runs strictly after
// this node). Global timer mutations are collected into pfc_state.want_*
// and flushed in port_id order by flushPortPfcTimers. Global events are
// pushed into the outbox for the follow-up flushPortOutbox singleton.
//
// The function body is a small dispatcher that forwards the heavy work to
// pfcDetectOnePortEgress / pfcDetectOnePortIngress. Keeping the two
// branches in separate MADRONA_NO_INLINE callees is what lets NVRTC +
// ptxas finish per-TU optimisation under `-dlto -dopt=on
// --extra-device-vectorization`; the monolithic form used to hang the
// compiler on this TU.
MADRONA_NO_INLINE void Sim::markPfcIngressCheckTargets(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        ctx.get<PortPfcState>(port_e).pfc_check_target = 0;
    }

    if (enablePfc == 0 || pfcEgress != 0) {
        return;
    }

    for (int32_t egress_port = 0; egress_port < numPorts; egress_port++) {
        Entity egress_e = portEntities[egress_port];
        if (egress_e == Entity::none() ||
            ctx.get<DirtyPort>(egress_e).isDirty == 0) {
            continue;
        }

        const PortTagList &tag_list = ctx.get<PortTagList>(egress_e);
        for (int32_t i = 0; i < tag_list.count; i++) {
            Entity tag_e = tag_list.tags[i];
            if (tag_e == Entity::none()) {
                continue;
            }

            const FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            int32_t ingress_port = tag.ingress_port_id;
            if (ingress_port < 0 || ingress_port >= numPorts) {
                continue;
            }

            Entity ingress_e = portEntities[ingress_port];
            if (ingress_e != Entity::none()) {
                ctx.get<PortPfcState>(ingress_e).pfc_check_target = 1;
                int32_t upstream_port = peerPort[ingress_port];
                if (flowWatchNodeEnabled(*this, upstream_port)) {
                    printFlowWatchPfcMark(systemLogStep, now,
                        ingress_port, portToNode[ingress_port],
                        upstream_port, portToNode[upstream_port],
                        egress_port, portToNode[egress_port], tag);
                }
            }
        }
    }
}

void Sim::pfcDetectOnePort(
    Context &ctx,
    int32_t port_id,
    PortState & /*port_state*/,
    PortBuffer & /*port_buf*/,
    DirtyPort &dirty,
    PortPfcConfig &cfg,
    PortPfcState &state,
    PortOutbox &outbox,
    PortTraceLast &trace,
    PortTagList &tag_list,
    IngressTagList &ingress_list)
{
    // Reset per-frame scratch for this port.
    outbox.num_events = 0;
    bool keep_trace = traceModeEnabled();
    if (keep_trace) {
        trace.pfc_detect_checked = 0;
        trace.pfc_detect_emitted = 0;
    }
    trace.pfc_detect_dropped = 0;
    state.want_clear_pause = 0;
    state.want_clear_resume = 0;
    state.want_set_pause = 0;
    state.want_set_resume = 0;
    state.set_pause_t = 0.0;
    state.set_resume_t = 0.0;

    if (enablePfc == 0) {
        return;
    }

    if (pfcEgress != 0) {
        pfcDetectOnePortEgress(ctx, port_id, dirty, cfg, state,
            outbox, trace, tag_list);
    } else {
        pfcDetectOnePortIngress(ctx, port_id, cfg, state, outbox, trace,
            ingress_list);
    }
}

MADRONA_NO_INLINE void Sim::pfcDetectOnePortEgress(
    Context &ctx,
    int32_t port_id,
    DirtyPort &dirty,
    PortPfcConfig &cfg,
    PortPfcState &state,
    PortOutbox &outbox,
    PortTraceLast &trace,
    PortTagList &tag_list)
{
    // -------- egress mode: this port is the egress port --------
    if (dirty.isDirty == 0) {
        return;
    }
    if (cfg.pfc_enabled == 0) {
        return;
    }
    if (traceModeEnabled()) {
        trace.pfc_detect_checked = 1;
    }

    double buf_by_pri[PFC_MAX_PRIORITY] {};
    int32_t upstream_ports[MAX_TOPO_PORTS] {};
    int32_t num_upstream_ports = 0;

    // Phase D: iterate this port's PortTagList rather than the global
    // tagIndex. Same semantics since PortTagList is authoritative for
    // tags whose FlowTagState.port_id == this port.
    for (int32_t i = 0; i < tag_list.count; i++) {
        Entity te = tag_list.tags[i];
        if (te == Entity::none()) {
            continue;
        }
        FlowTagState &tag = ctx.get<FlowTagState>(te);
        if (tag.is_source != 0) {
            continue;
        }
        materializeBacklog(tag, now);

        // Reconcile tag backlog with actual port buffer state (egress mode).
        // Same stagnation recovery as ingress mode.
        {
            Entity ep_e = portEntities[port_id];
            if (ep_e != Entity::none()) {
                PortBuffer &port_buf = ctx.get<PortBuffer>(ep_e);
                materializeBufCnt(port_buf, now);
                int32_t epri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                double actual = port_buf.prior_bufs[epri].buf_cnt;
                if (actual < 1.0 && tag.backlog > 1.0) {
                    tag.backlog = actual;
                    tag.last_backlog_time = now;
                }
            }
        }

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
                Time pfc_delay = getPortLinkDelay(port_id, up);
                if (outbox.num_events < MAX_PORT_OUTBOX) {
                    DelayedEvent &ev = outbox.events[outbox.num_events++];
                    ev = DelayedEvent {};
                    ev.t = computePropagationTime(pfc_delay);
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = up,
                        .source_port_id = port_id,
                        .priority = pri,
                        .paused = 1,
                    };
                    if (flowWatchNodeEnabled(*this, up)) {
                        printFlowWatchPfcEmit(systemLogStep, now, "egress_xoff",
                            port_id, portToNode[port_id], up, portToNode[up],
                            pri, 1, buf, 0.0);
                    }
                    if (traceModeEnabled()) {
                        trace.pfc_detect_emitted += 1;
                    }
                } else {
                    trace.pfc_detect_dropped += 1;
                }
            }
            state.pfc_cnt[pri] += 1;
        } else if (state.pause_active[pri] != 0 && buf <= cfg.xon[pri] + 0.5) {
            state.pause_active[pri] = 0;
            state_changed = true;
            for (int32_t j = 0; j < state.paused_upstream_count[pri]; j++) {
                int32_t up = state.paused_upstreams[pri][j];
                Time pfc_delay = getPortLinkDelay(port_id, up);
                if (outbox.num_events < MAX_PORT_OUTBOX) {
                    DelayedEvent &ev = outbox.events[outbox.num_events++];
                    ev = DelayedEvent {};
                    ev.t = computePropagationTime(pfc_delay);
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = up,
                        .source_port_id = port_id,
                        .priority = pri,
                        .paused = 0,
                    };
                    if (flowWatchNodeEnabled(*this, up)) {
                        printFlowWatchPfcEmit(systemLogStep, now, "egress_xon",
                            port_id, portToNode[port_id], up, portToNode[up],
                            pri, 0, buf, 0.0);
                    }
                    if (traceModeEnabled()) {
                        trace.pfc_detect_emitted += 1;
                    }
                } else {
                    trace.pfc_detect_dropped += 1;
                }
            }
            state.paused_upstream_count[pri] = 0;
        }
    }

    if (state_changed) {
        state.want_clear_pause = 1;
        state.want_clear_resume = 1;
    }
}

MADRONA_NO_INLINE void Sim::pfcDetectOnePortIngress(
    Context &ctx,
    int32_t port_id,
    PortPfcConfig &cfg,
    PortPfcState &state,
    PortOutbox &outbox,
    PortTraceLast &trace,
    IngressTagList &ingress_list)
{
    // -------- ingress mode: this port is the ingress port --------
    // markPfcIngressCheckTargets mirrors Jiuding's legacy ingress_check set:
    // scan dirty egress ports' live PortTagLists, then mark each contributing
    // ingress port exactly once before this per-port detect pass.
    if (state.pfc_check_target == 0) {
        return;
    }
    if (cfg.pfc_enabled == 0) {
        return;
    }
    if (traceModeEnabled()) {
        trace.pfc_detect_checked = 1;
    }

    double buf_by_pri[PFC_MAX_PRIORITY] {};
    double net_rate_by_pri[PFC_MAX_PRIORITY] {};
    for (int32_t i = 0; i < ingress_list.count; i++) {
        Entity te = ingress_list.tags[i];
        if (te == Entity::none()) {
            continue;
        }
        FlowTagState &tag = ctx.get<FlowTagState>(te);
        if (tag.port_entity == Entity::none()) {
            continue;
        }
        int32_t egress_port = tag.port_id;
        if (egress_port < 0 || egress_port >= numPorts ||
            tag.port_entity != portEntities[egress_port]) {
            continue;
        }
        if (tag.is_source != 0) {
            continue;
        }
        materializeBacklog(tag, now);

        // Reconcile tag backlog with actual port buffer state.
        // When both in_bw and out_bw are 0 (all paused), tag.backlog freezes
        // even though buf_cnt may have drained to 0. Cap tag backlogs to the
        // actual port buffer level so PFC can detect XON correctly.
        // Matches Jiuding's stagnation recovery in systems.cpp:687-778.
        Entity eport_e = portEntities[egress_port];
        if (eport_e != Entity::none()) {
            PortBuffer &port_buf = ctx.get<PortBuffer>(eport_e);
            materializeBufCnt(port_buf, now);
            int32_t eport_pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
            double actual = port_buf.prior_bufs[eport_pri].buf_cnt;
            if (actual < 1.0 && tag.backlog > 1.0) {
                tag.backlog = actual;
                tag.last_backlog_time = now;
            }
        }

        int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
        buf_by_pri[pri] += tag.backlog;
        net_rate_by_pri[pri] += (tag.in_bw - tag.out_bw);
    }

    int32_t upstream_port = -1;
    if (port_id >= 0 && port_id < numPorts) {
        upstream_port = peerPort[port_id];
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
                Time pfc_delay = getPortLinkDelay(port_id, upstream_port);
                if (outbox.num_events < MAX_PORT_OUTBOX) {
                    DelayedEvent &ev = outbox.events[outbox.num_events++];
                    ev = DelayedEvent {};
                    ev.t = computePropagationTime(pfc_delay);
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = upstream_port,
                        .source_port_id = port_id,
                        .priority = pri,
                        .paused = 1,
                    };
                    if (flowWatchNodeEnabled(*this, upstream_port)) {
                        printFlowWatchPfcEmit(systemLogStep, now,
                            "ingress_xoff", port_id, portToNode[port_id],
                            upstream_port, portToNode[upstream_port],
                            pri, 1, buf, net_rate_by_pri[pri]);
                    }
                    if (traceModeEnabled()) {
                        trace.pfc_detect_emitted += 1;
                    }
                } else {
                    trace.pfc_detect_dropped += 1;
                }
            }
            state.pfc_cnt[pri] += 1;
        } else if (state.pause_active[pri] != 0 && buf <= cfg.xon[pri] + 0.5) {
            state.pause_active[pri] = 0;
            state_changed = true;
            for (int32_t k = 0; k < state.paused_upstream_count[pri]; k++) {
                int32_t up = state.paused_upstreams[pri][k];
                Time pfc_delay = getPortLinkDelay(port_id, up);
                if (outbox.num_events < MAX_PORT_OUTBOX) {
                    DelayedEvent &ev = outbox.events[outbox.num_events++];
                    ev = DelayedEvent {};
                    ev.t = computePropagationTime(pfc_delay);
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = up,
                        .source_port_id = port_id,
                        .priority = pri,
                        .paused = 0,
                    };
                    if (flowWatchNodeEnabled(*this, up)) {
                        printFlowWatchPfcEmit(systemLogStep, now,
                            "ingress_xon", port_id, portToNode[port_id],
                            up, portToNode[up], pri, 0, buf,
                            net_rate_by_pri[pri]);
                    }
                    if (traceModeEnabled()) {
                        trace.pfc_detect_emitted += 1;
                    }
                } else {
                    trace.pfc_detect_dropped += 1;
                }
            }
            state.paused_upstream_count[pri] = 0;
        }
    }

    if (state_changed) {
        state.want_clear_pause = 1;
        state.want_clear_resume = 1;
    }

    // Rebuild timers from the post-transition state so a newly entered
    // pause period can immediately schedule its matching resume timer.
    // last-write-wins across priorities matches the legacy behaviour
    // (setPfcPauseTimer/setPfcResumeTimer index by port, not pri).
    //
    // When net_rate > 0 (upstream hasn't stopped yet due to link delay),
    // the resume timer uses effective_net = -out_total to predict when
    // the buffer would drain if only out_bw were active.  This matches
    // the legacy Jiuding logic.
    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
        double buf = buf_by_pri[pri];
        double net_rate = net_rate_by_pri[pri];
        if (state.pause_active[pri] == 0 && net_rate > 1.0) {
            double gap = cfg.xoff[pri] - buf;
            if (gap > 1e-9) {
                double t_xoff = gap / net_rate;
                if (t_xoff > 1e-9 && t_xoff < 1e6) {
                    state.want_set_pause = 1;
                    state.set_pause_t = t_xoff;
                }
            }
        } else if (state.pause_active[pri] != 0) {
            double effective_net = net_rate;
            if (effective_net >= -1e-15) {
                double out_total = 0.0;
                for (int32_t i = 0; i < ingress_list.count; i++) {
                    Entity te = ingress_list.tags[i];
                    if (te == Entity::none()) {
                        continue;
                    }
                    FlowTagState &t = ctx.get<FlowTagState>(te);
                    if (t.priority == pri) {
                        out_total += t.out_bw;
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
                        state.want_set_resume = 1;
                        state.set_resume_t = t_xon;
                    }
                }
            }
        }
    }
}

}
