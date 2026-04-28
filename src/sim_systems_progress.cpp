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

void Sim::progressFinishedSources(Context &ctx,
                                  Time dt,
                                  Time next_now,
                                  int32_t &finished_source_count,
                                  int32_t &emitted_cleanup_count)
{
    bool need_check_finish =
        cachedNextFinishTime < std::numeric_limits<Time>::max() &&
        cachedNextFinishTime <= dt + 1e-12;

    if (need_check_finish) {
        int32_t num_finished = 0;
        Time next_finish = std::numeric_limits<Time>::max();

        for (int32_t i = 0; i < numSourceTags; i++) {
            Entity tag_e = sourceTags[i].entity;
            if (tag_e == Entity::none()) {
                continue;
            }

            FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            materializeRemaining(tag, next_now);

            if (tag.remaining < 1.0) {
                tag.remaining = 0.0;
                if (num_finished < MAX_SOURCE_TAGS) {
                    finishedSourceScratch[num_finished++] = tag_e;
                }
                finished_source_count += 1;

                if (tag.next_port_id >= 0) {
                    Time link_delay = defaultLinkDelay;
                    int32_t src_node_slot =
                        findNodeSlot(portToNode[tag.port_id]);
                    int32_t dst_node_slot =
                        findNodeSlot(portToNode[tag.next_port_id]);

                    if (src_node_slot >= 0 && dst_node_slot >= 0 &&
                        linkDelays[src_node_slot][dst_node_slot] >= 0.0) {
                        link_delay = linkDelays[src_node_slot][dst_node_slot];
                    }

                    DelayedEvent ev {};
                    ev.t = next_now + link_delay;
                    ev.type = DelayedEvent::Type::BwUpdate;
                    ev.bwupd = BwUpdateEv {
                        .port_id = tag.next_port_id,
                        .flow_id = tag.flow_id,
                        .in_bw = 0.0,
                    };
                    pushDelayedEvent(ev);
                    emitted_cleanup_count += 1;
                }
            } else if (tag.out_bw > 1e-15) {
                Time t_finish = tag.remaining / tag.out_bw;
                if (t_finish > 1e-15) {
                    next_finish = std::min(next_finish, t_finish);
                }
            }
        }

        for (int32_t i = 0; i < num_finished; i++) {
            destroyTag(ctx, finishedSourceScratch[i], false, next_now);
        }

        cachedNextFinishTime = next_finish;
    } else if (cachedNextFinishTime < std::numeric_limits<Time>::max()) {
        cachedNextFinishTime -= dt;
        if (cachedNextFinishTime < 1e-15) {
            cachedNextFinishTime = 1e-15;
        }
    }

    if (cachedNextDrainTime < std::numeric_limits<Time>::max()) {
        cachedNextDrainTime -= dt;
        if (cachedNextDrainTime < 1e-15) {
            cachedNextDrainTime = timerInactiveSentinel();
            cachedDrainPortID = -1;
        }
    }
}

void Sim::progressBacklogDrainTimers(Context &ctx, Time dt)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        if (!timerIsActive(backlogDrainTimers[port_id])) {
            continue;
        }

        backlogDrainTimers[port_id] -= dt;
        if (backlogDrainTimers[port_id] < 1e-15) {
            Entity port_e = portEntities[port_id];
            if (port_e != Entity::none()) {
                portDirtyStates[port_id].isDirty = 1;
            }
            clearBacklogDrainTimer(port_id);
        }
    }
}

void Sim::markIngressTagsDirty(Context &ctx, int32_t ingress_port)
{
    for (int32_t tag_idx = 0; tag_idx < numIngressTags; tag_idx++) {
        if (ingressTags[tag_idx].ingress_port_id != ingress_port) {
            continue;
        }

        Entity tag_e = ingressTags[tag_idx].entity;
        if (tag_e == Entity::none()) {
            continue;
        }

        FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
        if (tag.port_id < 0 || tag.port_id >= numPorts) {
            continue;
        }

        Entity port_e = portEntities[tag.port_id];
        if (port_e != Entity::none()) {
            portDirtyStates[tag.port_id].isDirty = 1;
        }
    }
}

void Sim::progressPfcTimers(Context &ctx, Time dt)
{
    for (int32_t ingress_port = 0; ingress_port < numPorts; ingress_port++) {
        if (!timerIsActive(pfcPauseTimers[ingress_port])) {
            continue;
        }

        pfcPauseTimers[ingress_port] -= dt;
        if (pfcPauseTimers[ingress_port] < 1e-9) {
            markIngressTagsDirty(ctx, ingress_port);
            clearPfcPauseTimer(ingress_port);
        }
    }

    for (int32_t ingress_port = 0; ingress_port < numPorts; ingress_port++) {
        if (!timerIsActive(pfcResumeTimers[ingress_port])) {
            continue;
        }

        pfcResumeTimers[ingress_port] -= dt;
        if (pfcResumeTimers[ingress_port] < 1e-9) {
            markIngressTagsDirty(ctx, ingress_port);
            clearPfcResumeTimer(ingress_port);
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
                portDirtyStates[port_id].isDirty = 1;
                break;
            }
        }
    }
}

void Sim::progressExhaustedPfcState(Context &ctx, Time dt)
{
    for (int32_t ingress_port = 0; ingress_port < numPorts; ingress_port++) {
        Entity ingress_e = portEntities[ingress_port];
        if (ingress_e == Entity::none()) {
            continue;
        }

        PortPfcState &pfc = portPfcStates[ingress_port];
        if (!hasActivePause(pfc)) {
            continue;
        }

        bool no_tags = true;
        for (int32_t tag_idx = 0; tag_idx < numIngressTags; tag_idx++) {
            if (ingressTags[tag_idx].ingress_port_id == ingress_port &&
                ingressTags[tag_idx].entity != Entity::none()) {
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
                    pushDelayedEvent(ev);
                }

                pfc.paused_upstream_count[pri] = 0;
            }

            continue;
        }

        bool any_capped = false;
        for (int32_t tag_idx = 0; tag_idx < numIngressTags; tag_idx++) {
            if (ingressTags[tag_idx].ingress_port_id != ingress_port) {
                continue;
            }

            Entity tag_e = ingressTags[tag_idx].entity;
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

    progressFinishedSources(ctx, dt, next_now, finished_source_count,
        emitted_cleanup_count);
    progressBacklogDrainTimers(ctx, dt);
    progressPfcTimers(ctx, dt);

    bool all_exhausted =
        cachedNextDrainTime >= timerInactiveSentinel() &&
        cachedNextFinishTime >= timerInactiveSentinel() &&
        numDelayedEvents == 0 &&
        numPendingFlows == 0 &&
        !hasActiveBacklogDrainTimers() &&
        !hasActivePfcPauseTimers() &&
        !hasActivePfcResumeTimers();

    if (all_exhausted) {
        markBufferedPortsDirty(ctx);
        if (enablePfc != 0) {
            progressExhaustedPfcState(ctx, dt);
        }
    }

    if (log_enabled) {
        double next_finish_gap =
            cachedNextFinishTime < timerInactiveSentinel() ?
            cachedNextFinishTime :
            std::numeric_limits<double>::max();
        printSystemProgressSummary(step, now, dt, finished_source_count,
            emitted_cleanup_count, next_now, next_finish_gap);
    }
}

}
