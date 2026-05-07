#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

inline bool hasActivePause(const PortPfcState &pfc)
{
    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
        if (pfc.pause_active[pri] != 0) {
            return true;
        }
    }
    return false;
}

} // namespace

MADRONA_NO_INLINE void Sim::progressFinishedSourcesOnePort(
    Context &ctx,
    Time dt,
    Time next_now,
    int32_t port_id,
    PortState &,
    PortTagList &tag_list,
    const PortSourceTagList &source_tag_list,
    PortCachedHints &hints,
    PortFinishedSourceList &finished_list,
    PortTraceLast &trace,
    PortOutbox &outbox)
{
    trace.progress_source_scan_count = 0;
    trace.progress_finished_source_count = 0;
    trace.progress_emitted_cleanup_count = 0;
    hints.has_finish_hint = 0;
    hints.finish_hint_t = std::numeric_limits<Time>::max();
    finished_list.num = 0;
    outbox.num_events = 0;

    if (hints.has_active_finish == 0) {
        return;
    }

    hints.active_finish_t -= dt;
    if (hints.active_finish_t > 1e-12) {
        return;
    }

    hints.has_active_finish = 0;
    hints.active_finish_t = 0.0;

    auto handle_source_tag = [&](Entity tag_e) {
        trace.progress_source_scan_count += 1;
        FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
        materializeRemaining(tag, next_now);

        if (tag.remaining < 1.0) {
            tag.remaining = 0.0;
            FlowTagProgress &progress = ctx.get<FlowTagProgress>(tag_e);
            if (progress.pending_source_destroy == 0 &&
                finished_list.num < MAX_PORT_CLEANUP) {
                progress.pending_source_destroy = 1;
                finished_list.tags[finished_list.num++] = tag_e;
                trace.progress_finished_source_count += 1;
            }

            if (tag.next_port_id >= 0) {
                Time link_delay = defaultLinkDelay;
                int32_t src_node_slot = findNodeSlot(portToNode[tag.port_id]);
                int32_t dst_node_slot =
                    findNodeSlot(portToNode[tag.next_port_id]);

                if (src_node_slot >= 0 && dst_node_slot >= 0 &&
                    linkDelays[src_node_slot][dst_node_slot] >= 0.0) {
                    link_delay = linkDelays[src_node_slot][dst_node_slot];
                }

                if (outbox.num_events < MAX_PORT_OUTBOX) {
                    DelayedEvent ev {};
                    ev.t = next_now + link_delay;
                    ev.type = DelayedEvent::Type::BwUpdate;
                    ev.bwupd = BwUpdateEv {
                        .port_id = tag.next_port_id,
                        .flow_id = tag.flow_id,
                        .in_bw = 0.0,
                    };
                    outbox.events[outbox.num_events++] = ev;
                    trace.progress_emitted_cleanup_count += 1;
                }
            }
        } else if (tag.out_bw > 1e-15) {
            Time t_finish = tag.remaining / tag.out_bw;
            if (t_finish > 1e-15) {
                if (!hints.has_finish_hint || t_finish < hints.finish_hint_t) {
                    hints.has_finish_hint = 1;
                    hints.finish_hint_t = t_finish;
                }
            }
        }
    };

    // Fast path: progress only needs source tags. If the per-port source
    // mirror stayed within capacity, we can skip scanning non-source tags.
    // Ports that overflow this mirror fall back to PortTagList so behaviour
    // remains identical.
    if (source_tag_list.overflow == 0) {
        for (int32_t i = 0; i < source_tag_list.count; i++) {
            Entity tag_e = source_tag_list.tags[i];
            if (tag_e == Entity::none()) {
                continue;
            }

            handle_source_tag(tag_e);
        }
        return;
    }

    for (int32_t i = 0; i < tag_list.count; i++) {
        Entity tag_e = tag_list.tags[i];
        if (tag_e == Entity::none()) {
            continue;
        }

        FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
        if (tag.is_source == 0) {
            continue;
        }

        handle_source_tag(tag_e);
    }
}

void Sim::progressBacklogDrainTimers(Context &ctx, Time dt)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        PortTimers &timers = ctx.get<PortTimers>(port_e);
        if (!timerIsActive(timers.backlog_drain)) {
            continue;
        }

        timers.backlog_drain -= dt;
        if (timers.backlog_drain < 1e-15) {
            ctx.get<DirtyPort>(port_e).isDirty = 1;
            clearBacklogDrainTimer(timers);
        }
    }
}

void Sim::markIngressTagsDirty(Context &ctx, int32_t ingress_port)
{
    if (ingress_port < 0 || ingress_port >= numPorts) {
        return;
    }

    Entity ingress_e = portEntities[ingress_port];
    if (ingress_e == Entity::none()) {
        return;
    }

    const IngressTagList &itl = ctx.get<IngressTagList>(ingress_e);
    for (int32_t i = 0; i < itl.count; i++) {
        Entity tag_e = itl.tags[i];
        if (tag_e == Entity::none()) {
            continue;
        }

        FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
        if (tag.port_id < 0 || tag.port_id >= numPorts) {
            continue;
        }

        Entity port_e = portEntities[tag.port_id];
        if (port_e != Entity::none()) {
            ctx.get<DirtyPort>(port_e).isDirty = 1;
        }
    }
}

void Sim::progressPfcTimers(Context &ctx, Time dt)
{
    for (int32_t ingress_port = 0; ingress_port < numPorts; ingress_port++) {
        Entity ingress_e = portEntities[ingress_port];
        if (ingress_e == Entity::none()) {
            continue;
        }

        PortTimers &timers = ctx.get<PortTimers>(ingress_e);
        if (!timerIsActive(timers.pfc_pause)) {
            continue;
        }

        timers.pfc_pause -= dt;
        if (timers.pfc_pause < 1e-9) {
            markIngressTagsDirty(ctx, ingress_port);
            clearPfcPauseTimer(timers);
        }
    }

    for (int32_t ingress_port = 0; ingress_port < numPorts; ingress_port++) {
        Entity ingress_e = portEntities[ingress_port];
        if (ingress_e == Entity::none()) {
            continue;
        }

        PortTimers &timers = ctx.get<PortTimers>(ingress_e);
        if (!timerIsActive(timers.pfc_resume)) {
            continue;
        }

        timers.pfc_resume -= dt;
        if (timers.pfc_resume < 1e-9) {
            markIngressTagsDirty(ctx, ingress_port);
            clearPfcResumeTimer(timers);
        }
    }
}

void Sim::markBufferedPortsDirty(Context &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        PortBuffer &port_buf = ctx.get<PortBuffer>(port_e);
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            PriorityBuffer &pb = port_buf.prior_bufs[pri];
            if (pb.buf_cnt > 1e-15 && pb.num_chunks > 0) {
                ctx.get<DirtyPort>(port_e).isDirty = 1;
                break;
            }
        }
    }
}

int32_t countBufferedPortsMarkedDirty(Sim &sim, Context &ctx)
{
    int32_t count = 0;
    for (int32_t port_id = 0; port_id < sim.numPorts; port_id++) {
        Entity port_e = sim.portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        if (ctx.get<DirtyPort>(port_e).isDirty == 0) {
            continue;
        }

        PortBuffer &port_buf = ctx.get<PortBuffer>(port_e);
        bool has_buffer = false;
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            PriorityBuffer &pb = port_buf.prior_bufs[pri];
            if (pb.buf_cnt > 1e-15 && pb.num_chunks > 0) {
                has_buffer = true;
                break;
            }
        }

        if (has_buffer) {
            count += 1;
        }
    }

    return count;
}

void Sim::progressExhaustedPfcState(Context &ctx, Time dt)
{
    for (int32_t ingress_port = 0; ingress_port < numPorts; ingress_port++) {
        Entity ingress_e = portEntities[ingress_port];
        if (ingress_e == Entity::none()) {
            continue;
        }

        PortPfcState &pfc = ctx.get<PortPfcState>(ingress_e);
        if (!hasActivePause(pfc)) {
            continue;
        }

        const IngressTagList &ingress_list =
            ctx.get<IngressTagList>(ingress_e);
        bool no_tags = true;
        for (int32_t i = 0; i < ingress_list.count; i++) {
            if (ingress_list.tags[i] != Entity::none()) {
                no_tags = false;
                break;
            }
        }

        if (no_tags) {
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                if (pfc.pause_active[pri] == 0) {
                    continue;
                }

                pfc.pause_active[pri] = 0;
                for (int32_t up_idx = 0;
                     up_idx < pfc.paused_upstream_count[pri];
                     up_idx++) {
                    int32_t upstream_port = pfc.paused_upstreams[pri][up_idx];
                    if (upstream_port < 0 || upstream_port >= numPorts) {
                        continue;
                    }

                    int32_t detect_slot =
                        findNodeSlot(portToNode[ingress_port]);
                    int32_t upstream_slot =
                        findNodeSlot(portToNode[upstream_port]);
                    Time pfc_delay = defaultLinkDelay;

                    if (detect_slot >= 0 && upstream_slot >= 0 &&
                        linkDelays[detect_slot][upstream_slot] >= 0.0) {
                        pfc_delay =
                            linkDelays[detect_slot][upstream_slot];
                    }

                    DelayedEvent ev {};
                    ev.t = now + dt + pfc_delay;
                    ev.type = DelayedEvent::Type::PfcControl;
                    ev.pfcctrl = PfcControlEv {
                        .target_port_id = upstream_port,
                        .source_port_id = ingress_port,
                        .priority = pri,
                        .paused = 0,
                    };
                    pushDelayedEvent(ctx, ev);
                }

                pfc.paused_upstream_count[pri] = 0;
            }

            continue;
        }

        bool any_capped = false;
        for (int32_t i = 0; i < ingress_list.count; i++) {
            Entity tag_e = ingress_list.tags[i];
            if (tag_e == Entity::none()) {
                continue;
            }

            FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            if (tag.is_source != 0) {
                continue;
            }
            if (tag.port_id < 0 || tag.port_id >= numPorts) {
                continue;
            }

            Entity port_e = portEntities[tag.port_id];
            if (port_e == Entity::none()) {
                continue;
            }

            PortBuffer &port_buf = ctx.get<PortBuffer>(port_e);
            materializeBufCnt(port_buf, now);
            materializeBacklog(tag, now);

            int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
            double actual = port_buf.prior_bufs[pri].buf_cnt;

            if (actual < 1.0 && tag.backlog > 1.0) {
                tag.backlog = actual;
                tag.last_backlog_time = now;
                any_capped = true;
            }
        }

        if (any_capped) {
            markIngressTagsDirty(ctx, ingress_port);
        }
    }
}

void Sim::flowProgressAndCleanupSystem(Context &ctx, Time dt)
{
    constexpr const char *scope = "progress";
    uint64_t step = systemLogStep;
    bool log_enabled =
        traceModeEnabled() && compiledSystemLogEnabled(scope, step);
    Time next_now = now + dt;
    int32_t finished_source_count = 0;
    int32_t emitted_cleanup_count = 0;
    int32_t source_scan_count = 0;
    int32_t source_destroy_count = 0;
    int32_t buffered_dirty_port_count = 0;

    flushPortOutbox(ctx);
    const FlowCounters &flow_counters = ctx.singleton<FlowCounters>();
    SimRuntimeState &runtime = ctx.singleton<SimRuntimeState>();

    Time next_finish = timerInactiveSentinel();
    for (int32_t i = 0; i < numPorts; i++) {
        Entity port_e = portEntities[i];
        if (port_e == Entity::none()) {
            continue;
        }

        PortFinishedSourceList &finished_list =
            ctx.get<PortFinishedSourceList>(port_e);
        int32_t local_num = finished_list.num;
        Entity local_finished[MAX_PORT_CLEANUP] {};
        for (int32_t j = 0; j < local_num && j < MAX_PORT_CLEANUP; j++) {
            local_finished[j] = finished_list.tags[j];
            finished_list.tags[j] = Entity::none();
        }
        finished_list.num = 0;

        for (int32_t j = 0; j < local_num; j++) {
            Entity tag_e = local_finished[j];
            if (tag_e == Entity::none()) {
                continue;
            }

            FlowTagProgress &progress = ctx.get<FlowTagProgress>(tag_e);
            if (progress.pending_source_destroy == 0) {
                continue;
            }

            progress.pending_source_destroy = 0;
            destroyTag(ctx, tag_e, false, next_now);
            source_destroy_count += 1;
        }

        PortCachedHints &hints = ctx.get<PortCachedHints>(port_e);
        if (hints.has_active_finish != 0 &&
            hints.active_finish_t < next_finish) {
            next_finish = hints.active_finish_t;
        }
    }
    runtime.cachedNextFinishTime = next_finish;

    progressBacklogDrainTimers(ctx, dt);
    progressPfcTimers(ctx, dt);

    bool all_exhausted =
        runtime.cachedNextDrainTime >= timerInactiveSentinel() &&
        runtime.cachedNextFinishTime >= timerInactiveSentinel() &&
        runtime.numDelayedEvents == 0 &&
        flow_counters.numPendingFlows == 0 &&
        !hasActiveBacklogDrainTimers(ctx) &&
        !hasActivePfcPauseTimers(ctx) &&
        !hasActivePfcResumeTimers(ctx);

    if (all_exhausted) {
        markBufferedPortsDirty(ctx);
        buffered_dirty_port_count = countBufferedPortsMarkedDirty(*this, ctx);
        if (enablePfc != 0) {
            progressExhaustedPfcState(ctx, dt);
        }
    }

    if (log_enabled) {
        for (int32_t i = 0; i < numPorts; i++) {
            Entity port_e = portEntities[i];
            if (port_e == Entity::none()) {
                continue;
            }
            PortTraceLast &trace = ctx.get<PortTraceLast>(port_e);
            finished_source_count += trace.progress_finished_source_count;
            emitted_cleanup_count += trace.progress_emitted_cleanup_count;
            source_scan_count += trace.progress_source_scan_count;
        }
        double next_finish_gap =
            runtime.cachedNextFinishTime < timerInactiveSentinel() ?
            runtime.cachedNextFinishTime :
            std::numeric_limits<double>::max();
        printSystemProgressSummary(step, now, dt, finished_source_count,
            emitted_cleanup_count, source_scan_count, source_destroy_count,
            buffered_dirty_port_count, next_now, next_finish_gap);
    }
}

}
