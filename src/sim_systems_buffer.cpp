#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::materializeBacklog(FlowTagState &tag, Time at_time)
{
    if (tag.is_source != 0) {
        tag.backlog = 0.0;
        tag.last_backlog_time = at_time;
        return;
    }

    if (at_time > tag.last_backlog_time) {
        Time elapsed = at_time - tag.last_backlog_time;
        tag.backlog += (tag.in_bw - tag.out_bw) * elapsed;
        if (tag.backlog < 0.0) {
            tag.backlog = 0.0;
        }
        tag.last_backlog_time = at_time;
    }
}

void Sim::materializeRemaining(FlowTagState &tag, Time at_time)
{
    if (tag.is_source != 0 && at_time > tag.last_remaining_time) {
        Time elapsed = at_time - tag.last_remaining_time;
        tag.remaining -= tag.out_bw * elapsed;
        if (tag.remaining < 0.0) {
            tag.remaining = 0.0;
        }
        tag.last_remaining_time = at_time;
    }
}

double Sim::drainBufferChunks(PriorityBuffer &pb, double drain_bytes)
{
    if (drain_bytes <= 1e-15 || pb.num_chunks <= 0) {
        return 0.0;
    }

    double remaining = drain_bytes;
    double total = 0.0;

    while (remaining > 1e-15 && pb.num_chunks > 0) {
        BufferChunk &front = pb.buf_chunks[pb.head];
        if (front.chunk_bytes <= remaining + 1e-15) {
            remaining -= front.chunk_bytes;
            total += front.chunk_bytes;
            front = BufferChunk {};
            pb.head = (pb.head + 1) % MAX_BUFFER_CHUNKS;
            pb.num_chunks -= 1;
        } else {
            front.chunk_bytes -= remaining;
            total += remaining;
            remaining = 0.0;
        }
    }

    if (pb.num_chunks <= 0) {
        pb.head = 0;
        pb.tail = 0;
    }

    return total;
}

void Sim::alignChunksWithBufCnt(PriorityBuffer &pb)
{
    if (pb.num_chunks <= 0 || pb.buf_cnt < 1e-15) {
        return;
    }

    double chunk_sum = 0.0;
    for (int32_t i = 0; i < pb.num_chunks; i++) {
        int32_t idx = (pb.head + i) % MAX_BUFFER_CHUNKS;
        chunk_sum += pb.buf_chunks[idx].chunk_bytes;
    }

    double diff = pb.buf_cnt - chunk_sum;
    constexpr double EPS = 1.0;
    if (diff > EPS) {
        if (pb.num_chunks > 0 && pb.num_chunks < MAX_BUFFER_CHUNKS) {
            int32_t back_idx = (pb.tail - 1 + MAX_BUFFER_CHUNKS) % MAX_BUFFER_CHUNKS;
            BufferChunk gc {};
            gc.chunk_bytes = diff;
            gc.num_weights = pb.buf_chunks[back_idx].num_weights;
            for (int32_t i = 0; i < gc.num_weights; i++) {
                gc.weights[i] = pb.buf_chunks[back_idx].weights[i];
            }
            pb.buf_chunks[pb.tail] = gc;
            pb.tail = (pb.tail + 1) % MAX_BUFFER_CHUNKS;
            pb.num_chunks += 1;
        }
    } else if (diff < -EPS) {
        drainBufferChunks(pb, -diff);
        chunk_sum = 0.0;
        for (int32_t i = 0; i < pb.num_chunks; i++) {
            int32_t idx = (pb.head + i) % MAX_BUFFER_CHUNKS;
            chunk_sum += pb.buf_chunks[idx].chunk_bytes;
        }
        pb.buf_cnt = chunk_sum;
    }
}

void Sim::materializeBufCnt(PortBuffer &port_buf, Time at_time)
{
    if (at_time > port_buf.last_update_time + 1e-15) {
        Time elapsed = at_time - port_buf.last_update_time;
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            PriorityBuffer &pb = port_buf.prior_bufs[pri];
            pb.buf_cnt += pb.net_buffer_rate * elapsed;
            if (pb.buf_cnt < 0.0) {
                pb.buf_cnt = 0.0;
            }
        }
        port_buf.last_update_time = at_time;
    }
}

void Sim::bufferUpdateSystem(Context &ctx, Time dt)
{
    constexpr const char *scope = "buffer";
    uint64_t step = systemLogStep;
    bool log_enabled = systemLogEnabled(scope, step);

    if (enableBuffer == 0 || dt < 1e-15) {
        if (log_enabled) {
            printSystemBufferSummary(step, now, 0, 0, 0.0);
        }
        return;
    }

    bool processPorts[MAX_TOPO_PORTS] {};
    for (int32_t i = 0; i < numLastDirtyPortIDs; i++) {
        if (lastDirtyPortIDs[i] >= 0 && lastDirtyPortIDs[i] < numPorts) {
            processPorts[lastDirtyPortIDs[i]] = true;
        }
    }
    if (cachedDrainPortID >= 0 && cachedNextDrainTime < std::numeric_limits<Time>::max() && cachedNextDrainTime <= dt + 1e-12) {
        processPorts[cachedDrainPortID] = true;
    }
    for (int32_t i = 0; i < numBacklogDrainTimers; i++) {
        if (backlogDrainTimers[i] <= dt + 1e-12 && backlogDrainPortIDs[i] >= 0) {
            processPorts[backlogDrainPortIDs[i]] = true;
        }
    }

    constexpr double BUFFER_EPSILON = 1.0;
    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        if (processPorts[port_id]) {
            continue;
        }
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }
        PortBuffer &port_buf = ctx.get<PortBuffer>(port_e);
        bool any_pri_empty = false;
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            PriorityBuffer &pb = port_buf.prior_bufs[pri];
            if (pb.buf_cnt < 1e-15 && pb.num_chunks == 0 && std::abs(pb.net_buffer_rate) < 1e-15) {
                continue;
            }
            double eff = pb.buf_cnt;
            if (now > port_buf.last_update_time) {
                eff += pb.net_buffer_rate * (now - port_buf.last_update_time);
                if (eff < 0.0) {
                    eff = 0.0;
                }
            }
            if (eff < BUFFER_EPSILON) {
                any_pri_empty = true;
                break;
            }
        }
        if (any_pri_empty) {
            processPorts[port_id] = true;
        }
    }

    Time frame_end = now + dt;
    Entity tags_to_destroy[MAX_TAG_INDEX] {};
    int32_t num_destroy = 0;
    int32_t processed_port_count = 0;
    double total_buf_cnt = 0.0;

    for (int32_t port_id = 0; port_id < numPorts; port_id++) {
        if (!processPorts[port_id]) {
            continue;
        }
        Entity port_e = portEntities[port_id];
        if (port_e == Entity::none()) {
            continue;
        }

        PortBuffer &port_buf = ctx.get<PortBuffer>(port_e);
        materializeBufCnt(port_buf, now);

        Entity tags[MAX_TAG_INDEX] {};
        int32_t num_tags = 0;
        for (int32_t i = 0; i < numTagIndexEntries; i++) {
            if (tagIndex[i].port_id == port_id && tagIndex[i].entity != Entity::none()) {
                tags[num_tags++] = tagIndex[i].entity;
            }
        }
        if (num_tags == 0) {
            continue;
        }
        processed_port_count += 1;

        double port_bw = ctx.get<PortState>(port_e).port_bw;
        bool multi_pri = (qosMode == QOS_SP || qosMode == QOS_WRR);
        int32_t pri_begin = multi_pri ? 0 : 0;
        int32_t pri_end = multi_pri ? PFC_MAX_PRIORITY : 1;

        Entity pri_tags[PFC_MAX_PRIORITY][MAX_TAG_INDEX] {};
        int32_t pri_counts[PFC_MAX_PRIORITY] {};
        double pri_in[PFC_MAX_PRIORITY] {};
        double pri_out[PFC_MAX_PRIORITY] {};
        for (int32_t i = 0; i < num_tags; i++) {
            FlowTagState &tag = ctx.get<FlowTagState>(tags[i]);
            int32_t p = multi_pri ? std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1) : 0;
            pri_tags[p][pri_counts[p]++] = tags[i];
            pri_in[p] += (tag.is_source != 0) ? tag.out_bw : tag.in_bw;
            pri_out[p] += tag.out_bw;
        }

        for (int32_t pri = pri_begin; pri < pri_end; pri++) {
            if (pri_counts[pri] == 0 && pri_in[pri] < 1e-15 && pri_out[pri] < 1e-15) {
                continue;
            }

            PriorityBuffer &pb = port_buf.prior_bufs[pri];
            double in_total = pri_in[pri];
            double out_total = pri_out[pri];

            if (pb.buf_cnt > 1.0 && pb.num_chunks == 0) {
                BufferChunk comp {};
                comp.chunk_bytes = pb.buf_cnt;
                int32_t wcnt = 0;
                for (int32_t i = 0; i < pri_counts[pri] && wcnt < MAX_CHUNK_WEIGHTS; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                    comp.weights[wcnt].flow_id = tag.flow_id;
                    comp.weights[wcnt].weight = 1.0;
                    wcnt += 1;
                }
                comp.num_weights = wcnt;
                if (wcnt > 0) {
                    for (int32_t i = 0; i < wcnt; i++) {
                        comp.weights[i].weight /= (double)wcnt;
                    }
                    pb.buf_chunks[pb.tail] = comp;
                    pb.tail = (pb.tail + 1) % MAX_BUFFER_CHUNKS;
                    pb.num_chunks += 1;
                }
            }

            if (pb.buf_cnt > BUFFER_EPSILON && pb.num_chunks == 0) {
                BufferChunk comp {};
                comp.chunk_bytes = pb.buf_cnt;
                int32_t wcnt = 0;
                for (int32_t i = 0; i < pri_counts[pri] && wcnt < MAX_CHUNK_WEIGHTS; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                    materializeBacklog(tag, now);
                    if (tag.backlog > 1e-15) {
                        comp.weights[wcnt].flow_id = tag.flow_id;
                        comp.weights[wcnt].weight = 1.0;
                        wcnt += 1;
                    }
                }
                comp.num_weights = wcnt;
                if (wcnt > 0) {
                    for (int32_t i = 0; i < wcnt; i++) {
                        comp.weights[i].weight = 1.0 / (double)wcnt;
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
            bool has_existing_buffer = (pb.buf_cnt > 1e-15 && pb.num_chunks > 0);

            if (has_existing_buffer) {
                double buf_before = pb.buf_cnt;
                double add_amount = in_total * dt;
                if (add_amount > BUFFER_EPSILON && in_total > 1e-18) {
                    BufferChunk chunk {};
                    chunk.chunk_bytes = add_amount;
                    int32_t wcnt = 0;
                    for (int32_t i = 0; i < pri_counts[pri] && wcnt < MAX_CHUNK_WEIGHTS; i++) {
                        FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                        double effective_in = (tag.is_source != 0) ? tag.out_bw : tag.in_bw;
                        if (effective_in > 1e-18) {
                            chunk.weights[wcnt].flow_id = tag.flow_id;
                            chunk.weights[wcnt].weight = effective_in / in_total;
                            wcnt += 1;
                        }
                    }
                    chunk.num_weights = wcnt;
                    bool merged = false;
                    if (pb.num_chunks >= 2 && wcnt > 0) {
                        int32_t back_idx = (pb.tail - 1 + MAX_BUFFER_CHUNKS) % MAX_BUFFER_CHUNKS;
                        BufferChunk &back = pb.buf_chunks[back_idx];
                        if (back.num_weights == chunk.num_weights) {
                            merged = true;
                            for (int32_t i = 0; i < back.num_weights; i++) {
                                if (back.weights[i].flow_id != chunk.weights[i].flow_id ||
                                    std::abs(back.weights[i].weight - chunk.weights[i].weight) > 1e-6) {
                                    merged = false;
                                    break;
                                }
                            }
                        }
                        if (merged) {
                            back.chunk_bytes += add_amount;
                        }
                    }
                    if (!merged && wcnt > 0 && pb.num_chunks < MAX_BUFFER_CHUNKS) {
                        pb.buf_chunks[pb.tail] = chunk;
                        pb.tail = (pb.tail + 1) % MAX_BUFFER_CHUNKS;
                        pb.num_chunks += 1;
                    }
                    pb.buf_cnt += add_amount;
                }

                int32_t chunks_before = pb.num_chunks;
                double drain_amount = std::min(out_total * dt, pb.buf_cnt);
                if (drain_amount > BUFFER_EPSILON) {
                    double drained = drainBufferChunks(pb, drain_amount);
                    pb.buf_cnt = std::max(0.0, pb.buf_cnt - drained);
                }
                if (pb.num_chunks > 0) {
                    BufferChunk &front = pb.buf_chunks[pb.head];
                    if (front.chunk_bytes < BUFFER_EPSILON) {
                        pb.buf_cnt = std::max(0.0, pb.buf_cnt - front.chunk_bytes);
                        front = BufferChunk {};
                        pb.head = (pb.head + 1) % MAX_BUFFER_CHUNKS;
                        pb.num_chunks -= 1;
                    }
                }

                for (int32_t i = 0; i < pri_counts[pri]; i++) {
                    FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                    Time bl_elapsed = frame_end - tag.last_backlog_time;
                    if (bl_elapsed > 1e-15) {
                        double net = (tag.in_bw - tag.out_bw) * bl_elapsed;
                        tag.backlog = std::max(0.0, tag.backlog + net);
                    }
                    tag.last_backlog_time = frame_end;
                }

                bool chunk_consumed = pb.num_chunks < chunks_before;
                bool buffer_emptied = buf_before > 1e-15 && pb.buf_cnt < 1e-15;
                if (chunk_consumed || buffer_emptied) {
                    ctx.get<DirtyPort>(port_e).isDirty = 1;
                }

                if (chunk_consumed && !buffer_emptied) {
                    for (int32_t i = 0; i < pri_counts[pri]; i++) {
                        FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                        double reconciled = 0.0;
                        for (int32_t c = 0; c < pb.num_chunks; c++) {
                            int32_t idx = (pb.head + c) % MAX_BUFFER_CHUNKS;
                            BufferChunk &chunk = pb.buf_chunks[idx];
                            for (int32_t w = 0; w < chunk.num_weights; w++) {
                                if (chunk.weights[w].flow_id == tag.flow_id) {
                                    reconciled += chunk.chunk_bytes * chunk.weights[w].weight;
                                    break;
                                }
                            }
                        }
                        tag.backlog = reconciled;
                        tag.last_backlog_time = frame_end;
                    }
                }
                if (buffer_emptied) {
                    for (int32_t i = 0; i < pri_counts[pri]; i++) {
                        ctx.get<FlowTagState>(pri_tags[pri][i]).backlog = 0.0;
                    }
                }
            } else {
                double delta = (in_total - out_total) * dt;
                if (delta > BUFFER_EPSILON) {
                    pb.buf_cnt += delta;
                    BufferChunk chunk {};
                    chunk.chunk_bytes = delta;
                    int32_t wcnt = 0;
                    if (in_total > 1e-18) {
                        for (int32_t i = 0; i < pri_counts[pri] && wcnt < MAX_CHUNK_WEIGHTS; i++) {
                            FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                            double effective_in = (tag.is_source != 0) ? tag.out_bw : tag.in_bw;
                            if (effective_in > 1e-18) {
                                chunk.weights[wcnt].flow_id = tag.flow_id;
                                chunk.weights[wcnt].weight = effective_in / in_total;
                                if (tag.is_source == 0) {
                                    tag.backlog += delta * chunk.weights[wcnt].weight;
                                }
                                wcnt += 1;
                            }
                            tag.last_backlog_time = frame_end;
                        }
                    }
                    chunk.num_weights = wcnt;
                    if (wcnt > 0 && pb.num_chunks < MAX_BUFFER_CHUNKS) {
                        pb.buf_chunks[pb.tail] = chunk;
                        pb.tail = (pb.tail + 1) % MAX_BUFFER_CHUNKS;
                        pb.num_chunks += 1;
                    }
                    ctx.get<DirtyPort>(port_e).isDirty = 1;
                } else {
                    for (int32_t i = 0; i < pri_counts[pri]; i++) {
                        ctx.get<FlowTagState>(pri_tags[pri][i]).last_backlog_time = frame_end;
                    }
                }
            }

            for (int32_t i = 0; i < pri_counts[pri]; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(pri_tags[pri][i]);
                if (tag.in_bw != 0.0) {
                    continue;
                }
                bool upstream_alive = false;
                if (tag.is_source == 0) {
                    for (int32_t r = 0; r < numFlowRoutes; r++) {
                        if (flowRoutes[r].flow_id != tag.flow_id) {
                            continue;
                        }
                        for (int32_t s = 0; s < flowRoutes[r].num_steps; s++) {
                            if (flowRoutes[r].steps[s].next_port_id == tag.port_id) {
                                Entity up = findTag(flowRoutes[r].steps[s].port_id, tag.flow_id);
                                if (up != Entity::none()) {
                                    upstream_alive = true;
                                }
                                break;
                            }
                        }
                    }
                }
                if (tag.backlog < 1e-15) {
                    if (!upstream_alive && num_destroy < MAX_TAG_INDEX) {
                        tags_to_destroy[num_destroy++] = pri_tags[pri][i];
                    }
                    continue;
                }
                if (tag.out_bw == 0.0) {
                    bool pfc_paused = false;
                    if (enablePfc != 0) {
                        pfc_paused = ctx.get<PortPfcState>(port_e).paused[std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1)] != 0;
                    }
                    bool sp_starved = false;
                    if (qosMode != QOS_NONE) {
                        int32_t tp = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
                        PriorityBuffer &tpb = port_buf.prior_bufs[tp];
                        if (tpb.buf_cnt > 1e-15 || tpb.num_chunks > 0) {
                            sp_starved = true;
                        }
                    }
                    if (!pfc_paused && !upstream_alive && !sp_starved) {
                        tag.backlog = 0.0;
                        if (num_destroy < MAX_TAG_INDEX) {
                            tags_to_destroy[num_destroy++] = pri_tags[pri][i];
                        }
                    }
                }
            }
        }

        port_buf.last_update_time = frame_end;
        for (int32_t pri = pri_begin; pri < pri_end; pri++) {
            total_buf_cnt += port_buf.prior_bufs[pri].buf_cnt;
        }
    }

    for (int32_t i = 0; i < num_destroy; i++) {
        if (tags_to_destroy[i] != Entity::none()) {
            destroyTag(ctx, tags_to_destroy[i], true, frame_end);
        }
    }

    if (log_enabled) {
        printSystemBufferSummary(step, now, processed_port_count,
            num_destroy, total_buf_cnt);
    }
}

}
