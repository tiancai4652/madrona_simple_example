#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::portBandwidthAllocSystem(Engine &ctx, Time dt)
{
    (void)dt;

    constexpr const char *scope = "alloc";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);

    int32_t dirty_ports[MAX_TOPO_PORTS] {};
    int32_t num_dirty_ports = 0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        if (ctx.get<DirtyPort>(port_e).isDirty != 0) {
            dirty_ports[num_dirty_ports++] = port_id;
        }
    }

    if (num_dirty_ports == 0) {
        if (log_enabled) {
            printSystemAllocSummary(step, now, 0, 0, 0);
        }
        return;
    }

    int32_t processed_port_count = 0;
    int32_t dirty_tag_count = 0;

    for (int32_t i = 0; i < num_dirty_ports; i++) {
        int32_t pid = dirty_ports[i];
        if (pid == cachedDrainPortID) {
            cachedNextDrainTime = std::numeric_limits<Time>::max();
            cachedDrainPortID = -1;
        }
        clearBacklogDrainTimer(pid);
    }

    for (int32_t d = 0; d < num_dirty_ports; d++) {
        int32_t port_id = dirty_ports[d];
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        double port_bw = ctx.get<PortState>(port_e).port_bw;
        PortBuffer *port_buf = &ctx.get<PortBuffer>(port_e);
        PortPfcState *pfc_state = enablePfc ? &ctx.get<PortPfcState>(port_e) : nullptr;

        materializeBufCnt(*port_buf, now);
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            PriorityBuffer &pb = port_buf->prior_bufs[pri];
            if (pb.buf_cnt > 1.0 && pb.num_chunks <= 0) {
                BufferChunk comp {};
                comp.chunk_bytes = pb.buf_cnt;
                int32_t weight_count = 0;
                for (int32_t i = 0; i < numTagIndexEntries; i++) {
                    if (tagIndex[i].port_id != port_id) {
                        continue;
                    }
                    FlowTagState &tag = ctx.get<FlowTagState>(tagIndex[i].entity);
                    int p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                    if (p != pri) {
                        continue;
                    }
                    if (weight_count < MAX_CHUNK_WEIGHTS) {
                        comp.weights[weight_count].flow_id = tag.flow_id;
                        comp.weights[weight_count].weight = 1.0;
                        weight_count += 1;
                    }
                }
                comp.num_weights = weight_count;
                if (weight_count > 0) {
                    for (int32_t i = 0; i < weight_count; i++) {
                        comp.weights[i].weight = 1.0 / (double)weight_count;
                    }
                    pb.buf_chunks[pb.tail] = comp;
                    pb.tail = (pb.tail + 1) % MAX_BUFFER_CHUNKS;
                    pb.num_chunks += 1;
                }
            }
            if (pb.buf_cnt < 1e-15 && pb.num_chunks > 0) {
                double chunk_sum = 0.0;
                for (int32_t i = 0; i < pb.num_chunks; i++) {
                    int32_t idx = (pb.head + i) % MAX_BUFFER_CHUNKS;
                    chunk_sum += pb.buf_chunks[idx].chunk_bytes;
                }
                if (chunk_sum > 1e-15) {
                    pb.buf_cnt = chunk_sum;
                }
            }
            alignChunksWithBufCnt(pb);
        }

        Entity tags[MAX_TAG_INDEX] {};
        int32_t num_tags = 0;
        double sum_in = 0.0;
        for (int32_t i = 0; i < numTagIndexEntries; i++) {
            if (tagIndex[i].port_id != port_id) {
                continue;
            }
            Entity tag_e = tagIndex[i].entity;
            if (tag_e == Entity::none()) {
                continue;
            }
            tags[num_tags++] = tag_e;
            FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            sum_in += tag.in_bw;
        }
        if (num_tags == 0) {
            continue;
        }

        processed_port_count += 1;
        dirty_tag_count += num_tags;

        for (int32_t i = 0; i < num_tags; i++) {
            FlowTagState &tag = ctx.get<FlowTagState>(tags[i]);
            if (tag.is_source != 0) {
                materializeRemaining(tag, now);
            }
            materializeBacklog(tag, now);
            tag.prev_out_bw = tag.out_bw;
            tag.out_bw = 0.0;
        }

        for (int32_t i = 0; i < num_tags; i++) {
            FlowTagState &tag = ctx.get<FlowTagState>(tags[i]);
            if (tag.is_source == 0 && tag.in_bw == 0.0 && tag.backlog < 1.0) {
                tag.backlog = 0.0;
                destroyTag(ctx, tags[i], true, now);
                tags[i] = Entity::none();
            }
        }

        Entity live_tags[MAX_TAG_INDEX] {};
        int32_t num_live = 0;
        double live_sum_in = 0.0;
        for (int32_t i = 0; i < num_tags; i++) {
            if (tags[i] == Entity::none()) {
                continue;
            }
            live_tags[num_live++] = tags[i];
            FlowTagState &tag = ctx.get<FlowTagState>(tags[i]);
            live_sum_in += tag.in_bw;
        }
        if (num_live == 0) {
            continue;
        }

        double current_sum_in = live_sum_in;
        bool is_dest_only = false;
        if (qosMode != QOS_NONE) {
            int32_t node_slot = findNodeSlot(portToNode[port_id]);
            if (node_slot >= 0 && topoNodes[node_slot].type == NodeType::Host) {
                is_dest_only = true;
                for (int32_t i = 0; i < num_live; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                    if (tag.next_port_id >= 0) {
                        is_dest_only = false;
                        break;
                    }
                }
            }
        }

        double out_total = 0.0;
        if ((qosMode == QOS_SP || qosMode == QOS_WRR) && !is_dest_only) {
            Entity pri_tags[PFC_MAX_PRIORITY][MAX_TAG_INDEX] {};
            int32_t pri_counts[PFC_MAX_PRIORITY] {};
            double pri_in_sum[PFC_MAX_PRIORITY] {};
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                pri_tags[pri][pri_counts[pri]++] = live_tags[i];
                pri_in_sum[pri] += tag.in_bw;
            }

            if (qosMode == QOS_SP) {
                double remaining_bw = port_bw;
                for (int32_t pri = 0; pri < PFC_MAX_PRIORITY && remaining_bw > 1e-15; pri++) {
                    if (pri_counts[pri] == 0) {
                        continue;
                    }
                    if (pfc_state && pfc_state->paused[pri] != 0) {
                        continue;
                    }
                    PriorityBuffer &pb = port_buf->prior_bufs[pri];
                    bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                    double allocated = 0.0;
                    if (has_buf) {
                        BufferChunk &oldest = pb.buf_chunks[pb.head];
                        double active_weight_sum = 0.0;
                        for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                            Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                            if (t != Entity::none()) {
                                FlowTagState &tag = ctx.get<FlowTagState>(t);
                                if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                    active_weight_sum += oldest.weights[wi].weight;
                                }
                            }
                        }
                        if (active_weight_sum > 1e-15) {
                            for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                                Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                                if (t != Entity::none()) {
                                    FlowTagState &tag = ctx.get<FlowTagState>(t);
                                    if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                        tag.out_bw = remaining_bw * (oldest.weights[wi].weight / active_weight_sum);
                                        allocated += tag.out_bw;
                                    }
                                }
                            }
                        }
                        if (allocated < 1e-15) {
                            double bl_sum = 0.0;
                            for (int32_t i = 0; i < pri_counts[pri]; i++) {
                                FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                                double d = std::max(tag.in_bw, 0.0);
                                if (tag.backlog > 1e-15) {
                                    d += tag.backlog;
                                }
                                bl_sum += d;
                            }
                            if (bl_sum > 1e-15) {
                                for (int32_t i = 0; i < pri_counts[pri]; i++) {
                                    FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                                    double d = std::max(tag.in_bw, 0.0);
                                    if (tag.backlog > 1e-15) {
                                        d += tag.backlog;
                                    }
                                    tag.out_bw = remaining_bw * (d / bl_sum);
                                    allocated += tag.out_bw;
                                }
                            }
                        }
                    } else {
                        double demand = pri_in_sum[pri];
                        double alloc_bw = std::min(demand, remaining_bw);
                        if (demand > 1e-15) {
                            double scale = alloc_bw / demand;
                            for (int32_t i = 0; i < pri_counts[pri]; i++) {
                                FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                                tag.out_bw = tag.in_bw * scale;
                                allocated += tag.out_bw;
                            }
                        }
                    }
                    remaining_bw -= allocated;
                    if (remaining_bw < 0.0) {
                        remaining_bw = 0.0;
                    }
                }
            } else {
                bool active[PFC_MAX_PRIORITY] {};
                double weight_sum = 0.0;
                bool has_backlog[PFC_MAX_PRIORITY] {};
                double base_share[PFC_MAX_PRIORITY] {};
                double alloc[PFC_MAX_PRIORITY] {};
                double leftover = 0.0;
                for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                    if (pri_counts[pri] == 0) {
                        continue;
                    }
                    if (pfc_state && pfc_state->paused[pri] != 0) {
                        continue;
                    }
                    if (priorWeights[pri] < 1e-15) {
                        continue;
                    }
                    active[pri] = true;
                    weight_sum += priorWeights[pri];
                }
                if (weight_sum > 1e-15) {
                    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                        if (!active[pri]) {
                            continue;
                        }
                        base_share[pri] = port_bw * (priorWeights[pri] / weight_sum);
                        PriorityBuffer &pb = port_buf->prior_bufs[pri];
                        bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                        bool any_backlog = false;
                        for (int32_t i = 0; i < pri_counts[pri]; i++) {
                            if (ctx.get<FlowTagState>(pri_tags[pri][i]).backlog > 1e-15) {
                                any_backlog = true;
                                break;
                            }
                        }
                        has_backlog[pri] = has_buf || any_backlog;
                        if (has_backlog[pri]) {
                            alloc[pri] = base_share[pri];
                        } else {
                            double demand = pri_in_sum[pri];
                            alloc[pri] = std::min(base_share[pri], demand);
                            leftover += base_share[pri] - alloc[pri];
                        }
                    }
                    if (leftover > 1e-15) {
                        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                            if (!active[pri] || has_backlog[pri]) {
                                continue;
                            }
                            double demand = pri_in_sum[pri];
                            if (demand > alloc[pri]) {
                                double top_up = std::min(demand - alloc[pri], leftover);
                                alloc[pri] += top_up;
                                leftover -= top_up;
                                if (leftover < 1e-15) {
                                    break;
                                }
                            }
                        }
                        if (leftover > 1e-15) {
                            double bl_weight_sum = 0.0;
                            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                                if (active[pri] && has_backlog[pri]) {
                                    bl_weight_sum += priorWeights[pri];
                                }
                            }
                            if (bl_weight_sum > 1e-15) {
                                for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                                    if (active[pri] && has_backlog[pri]) {
                                        alloc[pri] += leftover * (priorWeights[pri] / bl_weight_sum);
                                    }
                                }
                            }
                        }
                    }
                    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                        if (!active[pri] || alloc[pri] < 1e-15) {
                            continue;
                        }
                        PriorityBuffer &pb = port_buf->prior_bufs[pri];
                        bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                        if (has_buf) {
                            BufferChunk &oldest = pb.buf_chunks[pb.head];
                            double aw = 0.0;
                            for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                                Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                                if (t != Entity::none()) {
                                    FlowTagState &tag = ctx.get<FlowTagState>(t);
                                    if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                        aw += oldest.weights[wi].weight;
                                    }
                                }
                            }
                            if (aw > 1e-15) {
                                for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                                    Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                                    if (t != Entity::none()) {
                                        FlowTagState &tag = ctx.get<FlowTagState>(t);
                                        if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                            tag.out_bw = alloc[pri] * (oldest.weights[wi].weight / aw);
                                        }
                                    }
                                }
                            }
                        } else {
                            double sum_pri_in = pri_in_sum[pri];
                            if (sum_pri_in > 1e-15) {
                                double scale = std::min(1.0, alloc[pri] / sum_pri_in);
                                for (int32_t i = 0; i < pri_counts[pri]; i++) {
                                    FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                                    tag.out_bw = tag.in_bw * scale;
                                }
                            }
                        }
                    }
                }
            }

            double pri_out_sum[PFC_MAX_PRIORITY] {};
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                out_total += tag.out_bw;
                int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                pri_out_sum[p] += tag.out_bw;
            }
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                port_buf->prior_bufs[pri].net_buffer_rate = 0.0;
            }
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                port_buf->prior_bufs[p].net_buffer_rate += tag.in_bw;
            }
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                port_buf->prior_bufs[pri].net_buffer_rate -= pri_out_sum[pri];
                PriorityBuffer &pb = port_buf->prior_bufs[pri];
                alignChunksWithBufCnt(pb);
                if (enableBuffer && pb.buf_cnt > 1e-15 && pb.num_chunks > 0) {
                    double out_pri = pri_out_sum[pri];
                    if (out_pri > 1e-6) {
                        BufferChunk &front = pb.buf_chunks[pb.head];
                        if (front.chunk_bytes > 1e-15) {
                            Time td = front.chunk_bytes / out_pri;
                            if (td > 1e-15 && td < cachedNextDrainTime) {
                                cachedNextDrainTime = td;
                                cachedDrainPortID = port_id;
                            }
                        }
                    }
                }
            }
        } else if (is_dest_only && qosMode != QOS_NONE) {
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                PriorityBuffer &pb = port_buf->prior_bufs[p];
                bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                bool needs_drain = tag.backlog > 1.0;
                if (has_buf || needs_drain) {
                    tag.out_bw = std::max(tag.in_bw, port_bw / (double)num_live);
                } else {
                    tag.out_bw = tag.in_bw;
                }
                out_total += tag.out_bw;
            }
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                double pi = 0.0;
                double po = 0.0;
                for (int32_t i = 0; i < num_live; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                    int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                    if (p == pri) {
                        pi += tag.in_bw;
                        po += tag.out_bw;
                    }
                }
                port_buf->prior_bufs[pri].net_buffer_rate = pi - po;
            }
        } else {
            PriorityBuffer &pb = port_buf->prior_bufs[0];
            bool has_buffer = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
            if (!has_buffer && live_sum_in < 1e-18) {
                continue;
            }
            bool is_congested = live_sum_in >= port_bw;
            if (has_buffer) {
                BufferChunk &oldest = pb.buf_chunks[pb.head];
                double active_weight_sum = 0.0;
                for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                    Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                    if (t != Entity::none()) {
                        FlowTagState &tag = ctx.get<FlowTagState>(t);
                        if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                            active_weight_sum += oldest.weights[wi].weight;
                        }
                    }
                }
                if (active_weight_sum > 1e-15) {
                    for (int32_t wi = 0; wi < oldest.num_weights; wi++) {
                        Entity t = findTag(port_id, oldest.weights[wi].flow_id);
                        if (t != Entity::none()) {
                            FlowTagState &tag = ctx.get<FlowTagState>(t);
                            if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
                                tag.out_bw = port_bw * (oldest.weights[wi].weight / active_weight_sum);
                            }
                        }
                    }
                } else if (live_sum_in > 1e-15) {
                    for (int32_t i = 0; i < num_live; i++) {
                        FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                        tag.out_bw = port_bw * (tag.in_bw / live_sum_in);
                    }
                }
            } else {
                if (live_sum_in >= 1e-18) {
                    if (is_congested) {
                        for (int32_t i = 0; i < num_live; i++) {
                            FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                            tag.out_bw = tag.in_bw * (port_bw / live_sum_in);
                        }
                    } else {
                        for (int32_t i = 0; i < num_live; i++) {
                            FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                            tag.out_bw = tag.in_bw;
                        }
                    }
                }
            }
            if (pfc_state) {
                for (int32_t i = 0; i < num_live; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                    if (pfc_state->paused[std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1)] != 0) {
                        tag.out_bw = 0.0;
                    }
                }
            }
            for (int32_t i = 0; i < num_live; i++) {
                out_total += ctx.get<FlowTagState>(live_tags[i]).out_bw;
            }
            if (enableBuffer && has_buffer) {
                BufferChunk &front = pb.buf_chunks[pb.head];
                if (port_bw > 1e-6 && front.chunk_bytes > 1e-15) {
                    Time td = front.chunk_bytes / port_bw;
                    if (td > 1e-15 && td < cachedNextDrainTime) {
                        cachedNextDrainTime = td;
                        cachedDrainPortID = port_id;
                    }
                }
            }
            pb.net_buffer_rate = live_sum_in - out_total;
        }

        if (log_enabled) {
            bool has_buffer = false;
            if (port_buf != nullptr) {
                if (qosMode == QOS_SP || qosMode == QOS_WRR) {
                    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                        PriorityBuffer &pb = port_buf->prior_bufs[pri];
                        if (pb.buf_cnt > 1e-15 && pb.num_chunks > 0) {
                            has_buffer = true;
                            break;
                        }
                    }
                } else {
                    PriorityBuffer &pb = port_buf->prior_bufs[0];
                    has_buffer = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
                }
            }
            printSystemAllocPort(step, now, port_id, port_bw, num_tags, num_live,
                current_sum_in, out_total, qosMode, is_dest_only ? 1 : 0,
                has_buffer ? 1 : 0);
        }

        for (int32_t i = 0; i < num_live; i++) {
            FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
            if (tag.is_source != 0) {
                if (tag.remaining > 0.0 && tag.remaining < 1.0) {
                    tag.remaining = 0.0;
                }
                if (tag.out_bw > 1e-15 && tag.remaining > 0.0) {
                    Time t_finish = tag.remaining / tag.out_bw;
                    if (t_finish > 0.0 && t_finish < cachedNextFinishTime) {
                        cachedNextFinishTime = t_finish;
                    }
                }
            }
            if (tag.is_source == 0 && tag.in_bw == 0.0 && tag.backlog > 1e-15 && tag.out_bw > 1e-15) {
                Time t_bl_drain = tag.backlog / tag.out_bw;
                if (t_bl_drain > 1e-15 && t_bl_drain < 1e6) {
                    int32_t idx = findBacklogDrainTimerIndex(port_id);
                    if (idx < 0 || t_bl_drain < backlogDrainTimers[idx]) {
                        setBacklogDrainTimer(port_id, t_bl_drain);
                    }
                }
            }
        }
    }

    if (log_enabled) {
        printSystemAllocSummary(step, now, num_dirty_ports,
            processed_port_count, dirty_tag_count);
    }
}

void Sim::downstreamEmitSystem(Engine &ctx)
{
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none() || ctx.get<DirtyPort>(port_e).isDirty == 0) {
            continue;
        }

        for (int32_t i = 0; i < numTagIndexEntries; i++) {
            if (tagIndex[i].port_id != port_id) {
                continue;
            }
            Entity tag_e = tagIndex[i].entity;
            if (tag_e == Entity::none()) {
                continue;
            }
            FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
            if (tag.next_port_id < 0) {
                continue;
            }
            if (tag.downstream_created == 0) {
                if (tag.out_bw > 1e-15) {
                    DelayedEvent ev {};
                    ev.t = computePropagationTimeForPort(tag.port_id, tag.next_port_id);
                    ev.type = DelayedEvent::Type::Arrival;
                    ev.arrival = FlowArrivalEv {
                        .port_id = tag.next_port_id,
                        .flow_id = tag.flow_id,
                        .size = 0.0,
                        .in_bw = tag.out_bw,
                        .is_source = 0,
                        .priority = tag.priority,
                    };
                    pushDelayedEvent(ev);
                    tag.downstream_created = 1;
                }
                continue;
            }
            if (tag.out_bw == tag.prev_out_bw) {
                continue;
            }
            DelayedEvent ev {};
            ev.t = computePropagationTimeForPort(tag.port_id, tag.next_port_id);
            ev.type = DelayedEvent::Type::BwUpdate;
            ev.bwupd = BwUpdateEv {
                .port_id = tag.next_port_id,
                .flow_id = tag.flow_id,
                .in_bw = tag.out_bw,
            };
            pushDelayedEvent(ev);
        }
    }
}

}
