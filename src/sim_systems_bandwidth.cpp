#include "sim.hpp"
#include "sim_debug.hpp"

#include <algorithm>
#include <limits>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

namespace {

struct PriorityTagBuckets {
    Entity tags[PFC_MAX_PRIORITY][MAX_TAGS_PER_PORT] {};
    int32_t counts[PFC_MAX_PRIORITY] {};
    double in_sum[PFC_MAX_PRIORITY] {};
};

struct AllocTagScratch {
    Entity tags[MAX_TAGS_PER_PORT] {};
    Entity live_tags[MAX_TAGS_PER_PORT] {};
    int32_t num_tags = 0;
    int32_t num_live = 0;
    double live_sum_in = 0.0;
};

MADRONA_NO_INLINE void resetAllocPhaseScratch(
    PortCachedHints &hints,
    PortDrainHint &drain_hint,
    PortCleanup &cleanup,
    PortTraceLast &trace)
{
    hints.has_drain_hint = 0;
    hints.has_finish_hint = 0;
    hints.drain_hint_t = 0.0;
    hints.finish_hint_t = 0.0;
    drain_hint.want_clear = 0;
    drain_hint.want_set = 0;
    drain_hint.set_t = 0.0;
    cleanup.num = 0;
    trace.has_alloc_trace = 0;
    trace.alloc_port_bw = 0.0;
    trace.alloc_num_tags = 0;
    trace.alloc_num_live = 0;
    trace.alloc_sum_in = 0.0;
    trace.alloc_sum_out = 0.0;
    trace.alloc_is_dest_only = 0;
    trace.alloc_has_buffer = 0;
    trace.was_dirty_at_alloc = 0;
}

MADRONA_NO_INLINE void materializeAllocPortBuffer(
    Sim &sim,
    Context &ctx,
    PortBuffer &port_buf,
    const PortTagList &tag_list)
{
    sim.materializeBufCnt(port_buf, sim.now);
    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
        PriorityBuffer &pb = port_buf.prior_bufs[pri];
        if (pb.buf_cnt > 1.0 && pb.num_chunks <= 0) {
            BufferChunk comp {};
            comp.chunk_bytes = pb.buf_cnt;
            int32_t weight_count = 0;
            for (int32_t i = 0; i < tag_list.count; i++) {
                Entity te = tag_list.tags[i];
                if (te == Entity::none()) {
                    continue;
                }
                FlowTagState &tag = ctx.get<FlowTagState>(te);
                int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
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
        sim.alignChunksWithBufCnt(pb);
    }
}

MADRONA_NO_INLINE void collectAllocTagScratch(
    Sim &sim,
    Context &ctx,
    const PortTagList &tag_list,
    PortCleanup &cleanup,
    AllocTagScratch &scratch)
{
    for (int32_t i = 0; i < tag_list.count; i++) {
        Entity tag_e = tag_list.tags[i];
        if (tag_e == Entity::none()) {
            continue;
        }
        scratch.tags[scratch.num_tags++] = tag_e;
    }

    for (int32_t i = 0; i < scratch.num_tags; i++) {
        FlowTagState &tag = ctx.get<FlowTagState>(scratch.tags[i]);
        if (tag.is_source != 0) {
            sim.materializeRemaining(tag, sim.now);
        }
        sim.materializeBacklog(tag, sim.now);
        tag.prev_out_bw = tag.out_bw;
        tag.out_bw = 0.0;
    }

    for (int32_t i = 0; i < scratch.num_tags; i++) {
        FlowTagState &tag = ctx.get<FlowTagState>(scratch.tags[i]);
        if (tag.is_source == 0 && tag.in_bw == 0.0 && tag.backlog < 1.0) {
            tag.backlog = 0.0;
            if (cleanup.num < MAX_PORT_CLEANUP) {
                cleanup.tags[cleanup.num] = scratch.tags[i];
                cleanup.propagate[cleanup.num] = 1;
                cleanup.num += 1;
            }
            scratch.tags[i] = Entity::none();
        }
    }

    for (int32_t i = 0; i < scratch.num_tags; i++) {
        Entity tag_e = scratch.tags[i];
        if (tag_e == Entity::none()) {
            continue;
        }
        scratch.live_tags[scratch.num_live++] = tag_e;
        scratch.live_sum_in += ctx.get<FlowTagState>(tag_e).in_bw;
    }
}

MADRONA_NO_INLINE bool detectDestOnlyAllocPort(
    Sim &sim,
    Context &ctx,
    int32_t port_id,
    Entity *live_tags,
    int32_t num_live)
{
    if (sim.qosMode == QOS_NONE) {
        return false;
    }

    int32_t node_slot = sim.findNodeSlot(sim.portToNode[port_id]);
    if (node_slot < 0 || sim.topoNodes[node_slot].type != NodeType::Host) {
        return false;
    }

    for (int32_t i = 0; i < num_live; i++) {
        if (ctx.get<FlowTagState>(live_tags[i]).next_port_id >= 0) {
            return false;
        }
    }

    return true;
}

MADRONA_NO_INLINE bool allocTraceHasBuffer(
    const Sim &sim,
    const PortBuffer &port_buf)
{
    if (sim.qosMode == QOS_SP || sim.qosMode == QOS_WRR) {
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            const PriorityBuffer &pb = port_buf.prior_bufs[pri];
            if (pb.buf_cnt > 1e-15 && pb.num_chunks > 0) {
                return true;
            }
        }
        return false;
    }

    const PriorityBuffer &pb = port_buf.prior_bufs[0];
    return pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
}

MADRONA_NO_INLINE void finalizeAllocPortState(
    Sim &sim,
    Context &ctx,
    PortBuffer &port_buf,
    PortCachedHints &hints,
    PortDrainHint &drain_hint,
    PortTraceLast &trace,
    Entity *live_tags,
    int32_t num_live,
    double port_bw,
    double sum_in,
    double out_total,
    bool is_dest_only)
{
    trace.has_alloc_trace = 1;
    trace.alloc_port_bw = port_bw;
    trace.alloc_sum_in = sum_in;
    trace.alloc_sum_out = out_total;
    trace.alloc_is_dest_only = is_dest_only ? 1 : 0;
    trace.alloc_has_buffer = allocTraceHasBuffer(sim, port_buf) ? 1 : 0;

    for (int32_t i = 0; i < num_live; i++) {
        FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
        if (tag.is_source != 0) {
            if (tag.remaining > 0.0 && tag.remaining < 1.0) {
                tag.remaining = 0.0;
            }
            if (tag.out_bw > 1e-15 && tag.remaining > 0.0) {
                Time t_finish = tag.remaining / tag.out_bw;
                if (t_finish > 0.0 &&
                    (!hints.has_finish_hint ||
                     t_finish < hints.finish_hint_t)) {
                    hints.has_finish_hint = 1;
                    hints.finish_hint_t = t_finish;
                }
            }
        }

        if (tag.is_source == 0 && tag.in_bw == 0.0 &&
            tag.backlog > 1e-15 && tag.out_bw > 1e-15) {
            Time t_bl_drain = tag.backlog / tag.out_bw;
            if (t_bl_drain > 1e-15 && t_bl_drain < 1e6 &&
                (!drain_hint.want_set || t_bl_drain < drain_hint.set_t)) {
                drain_hint.want_set = 1;
                drain_hint.set_t = t_bl_drain;
            }
        }
    }
}

MADRONA_NO_INLINE void collectPriorityTagBuckets(
    Context &ctx,
    Entity *live_tags,
    int32_t num_live,
    PriorityTagBuckets &buckets)
{
    for (int32_t i = 0; i < num_live; i++) {
        FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
        int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
        buckets.tags[pri][buckets.counts[pri]++] = live_tags[i];
        buckets.in_sum[pri] += tag.in_bw;
    }
}

MADRONA_NO_INLINE double activeChunkWeightSum(
    Sim &sim,
    Context &ctx,
    int32_t port_id,
    const BufferChunk &chunk)
{
    double active_weight_sum = 0.0;
    for (int32_t wi = 0; wi < chunk.num_weights; wi++) {
        Entity tag_e = sim.findTag(port_id, chunk.weights[wi].flow_id);
        if (tag_e == Entity::none()) {
            continue;
        }

        FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
        if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
            active_weight_sum += chunk.weights[wi].weight;
        }
    }

    return active_weight_sum;
}

MADRONA_NO_INLINE bool assignChunkWeightedOutBW(
    Sim &sim,
    Context &ctx,
    int32_t port_id,
    const BufferChunk &chunk,
    double alloc_bw)
{
    double active_weight_sum =
        activeChunkWeightSum(sim, ctx, port_id, chunk);
    if (active_weight_sum <= 1e-15) {
        return false;
    }

    for (int32_t wi = 0; wi < chunk.num_weights; wi++) {
        Entity tag_e = sim.findTag(port_id, chunk.weights[wi].flow_id);
        if (tag_e == Entity::none()) {
            continue;
        }

        FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
        if (tag.in_bw > 1e-15 || tag.backlog > 1e-15) {
            tag.out_bw = alloc_bw *
                (chunk.weights[wi].weight / active_weight_sum);
        }
    }

    return true;
}

MADRONA_NO_INLINE void finalizeQoSAlloc(
    Sim &sim,
    Context &ctx,
    PortBuffer &port_buf,
    Entity *live_tags,
    int32_t num_live,
    PortCachedHints &hints,
    double &out_total)
{
    double pri_out_sum[PFC_MAX_PRIORITY] {};
    for (int32_t i = 0; i < num_live; i++) {
        FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
        out_total += tag.out_bw;
        int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
        pri_out_sum[pri] += tag.out_bw;
    }

    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
        port_buf.prior_bufs[pri].net_buffer_rate = 0.0;
    }
    for (int32_t i = 0; i < num_live; i++) {
        FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
        int32_t pri = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
        port_buf.prior_bufs[pri].net_buffer_rate += tag.in_bw;
    }
    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
        port_buf.prior_bufs[pri].net_buffer_rate -= pri_out_sum[pri];
        PriorityBuffer &pb = port_buf.prior_bufs[pri];
        sim.alignChunksWithBufCnt(pb);
        if (!sim.enableBuffer || pb.buf_cnt <= 1e-15 || pb.num_chunks <= 0) {
            continue;
        }

        double out_pri = pri_out_sum[pri];
        if (out_pri <= 1e-6) {
            continue;
        }

        BufferChunk &front = pb.buf_chunks[pb.head];
        if (front.chunk_bytes <= 1e-15) {
            continue;
        }

        Time td = front.chunk_bytes / out_pri;
        if (td > 1e-15 &&
            (!hints.has_drain_hint || td < hints.drain_hint_t)) {
            hints.has_drain_hint = 1;
            hints.drain_hint_t = td;
        }
    }
}

// Split allocOnePort's heavy QoS branches into separate non-inlined helpers.
// The goal is to reduce the optimizer's control-flow explosion in one giant
// function body while keeping behaviour unchanged.
MADRONA_NO_INLINE void allocOnePortQoSSP(
    Sim &sim,
    Context &ctx,
    int32_t port_id,
    double port_bw,
    PortBuffer &port_buf,
    PortPfcState *pfc_state,
    Entity *live_tags,
    int32_t num_live,
    PortCachedHints &hints,
    double &out_total)
{
    PriorityTagBuckets buckets {};
    collectPriorityTagBuckets(ctx, live_tags, num_live, buckets);

    double remaining_bw = port_bw;
    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY && remaining_bw > 1e-15;
         pri++) {
        if (buckets.counts[pri] == 0) {
            continue;
        }
        if (pfc_state && pfc_state->paused[pri] != 0) {
            continue;
        }
        PriorityBuffer &pb = port_buf.prior_bufs[pri];
        bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
        double allocated = 0.0;
        if (has_buf) {
            BufferChunk &oldest = pb.buf_chunks[pb.head];
            if (assignChunkWeightedOutBW(
                    sim, ctx, port_id, oldest, remaining_bw)) {
                allocated = remaining_bw;
            }
            if (allocated < 1e-15) {
                double bl_sum = 0.0;
                for (int32_t i = 0; i < buckets.counts[pri]; i++) {
                    FlowTagState &tag =
                        ctx.get<FlowTagState>(buckets.tags[pri][i]);
                    double d = std::max(tag.in_bw, 0.0);
                    if (tag.backlog > 1e-15) {
                        d += tag.backlog;
                    }
                    bl_sum += d;
                }
                if (bl_sum > 1e-15) {
                    for (int32_t i = 0; i < buckets.counts[pri]; i++) {
                        FlowTagState &tag =
                            ctx.get<FlowTagState>(buckets.tags[pri][i]);
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
            double demand = buckets.in_sum[pri];
            double alloc_bw = std::min(demand, remaining_bw);
            if (demand > 1e-15) {
                double scale = alloc_bw / demand;
                for (int32_t i = 0; i < buckets.counts[pri]; i++) {
                    FlowTagState &tag =
                        ctx.get<FlowTagState>(buckets.tags[pri][i]);
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

    finalizeQoSAlloc(sim, ctx, port_buf, live_tags, num_live, hints, out_total);
}

MADRONA_NO_INLINE void allocOnePortQoSWRR(
    Sim &sim,
    Context &ctx,
    int32_t port_id,
    double port_bw,
    PortBuffer &port_buf,
    PortPfcState *pfc_state,
    Entity *live_tags,
    int32_t num_live,
    PortCachedHints &hints,
    double &out_total)
{
    PriorityTagBuckets buckets {};
    collectPriorityTagBuckets(ctx, live_tags, num_live, buckets);

    bool active[PFC_MAX_PRIORITY] {};
    double weight_sum = 0.0;
    bool has_backlog[PFC_MAX_PRIORITY] {};
    double base_share[PFC_MAX_PRIORITY] {};
    double alloc[PFC_MAX_PRIORITY] {};
    double leftover = 0.0;
    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
        if (buckets.counts[pri] == 0) {
            continue;
        }
        if (pfc_state && pfc_state->paused[pri] != 0) {
            continue;
        }
        if (sim.priorWeights[pri] < 1e-15) {
            continue;
        }
        active[pri] = true;
        weight_sum += sim.priorWeights[pri];
    }
    if (weight_sum > 1e-15) {
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            if (!active[pri]) {
                continue;
            }
            base_share[pri] = port_bw * (sim.priorWeights[pri] / weight_sum);
            PriorityBuffer &pb = port_buf.prior_bufs[pri];
            bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
            bool any_backlog = false;
            for (int32_t i = 0; i < buckets.counts[pri]; i++) {
                if (ctx.get<FlowTagState>(buckets.tags[pri][i]).backlog > 1e-15) {
                    any_backlog = true;
                    break;
                }
            }
            has_backlog[pri] = has_buf || any_backlog;
            if (has_backlog[pri]) {
                alloc[pri] = base_share[pri];
            } else {
                double demand = buckets.in_sum[pri];
                alloc[pri] = std::min(base_share[pri], demand);
                leftover += base_share[pri] - alloc[pri];
            }
        }
        if (leftover > 1e-15) {
            for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                if (!active[pri] || has_backlog[pri]) {
                    continue;
                }
                double demand = buckets.in_sum[pri];
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
                        bl_weight_sum += sim.priorWeights[pri];
                    }
                }
                if (bl_weight_sum > 1e-15) {
                    for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
                        if (active[pri] && has_backlog[pri]) {
                            alloc[pri] +=
                                leftover *
                                (sim.priorWeights[pri] / bl_weight_sum);
                        }
                    }
                }
            }
        }
        for (int32_t pri = 0; pri < PFC_MAX_PRIORITY; pri++) {
            if (!active[pri] || alloc[pri] < 1e-15) {
                continue;
            }
            PriorityBuffer &pb = port_buf.prior_bufs[pri];
            bool has_buf = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
            if (has_buf) {
                BufferChunk &oldest = pb.buf_chunks[pb.head];
                assignChunkWeightedOutBW(
                    sim, ctx, port_id, oldest, alloc[pri]);
            } else {
                double sum_pri_in = buckets.in_sum[pri];
                if (sum_pri_in > 1e-15) {
                    double scale = std::min(1.0, alloc[pri] / sum_pri_in);
                    for (int32_t i = 0; i < buckets.counts[pri]; i++) {
                        FlowTagState &tag =
                            ctx.get<FlowTagState>(buckets.tags[pri][i]);
                        tag.out_bw = tag.in_bw * scale;
                    }
                }
            }
        }
    }

    finalizeQoSAlloc(sim, ctx, port_buf, live_tags, num_live, hints, out_total);
}

MADRONA_NO_INLINE void allocOnePortDestOnly(
    Context &ctx,
    double port_bw,
    PortBuffer &port_buf,
    Entity *live_tags,
    int32_t num_live,
    double &out_total)
{
    for (int32_t i = 0; i < num_live; i++) {
        FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
        int32_t p = std::clamp(tag.priority, 0, PFC_MAX_PRIORITY - 1);
        PriorityBuffer &pb = port_buf.prior_bufs[p];
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
        port_buf.prior_bufs[pri].net_buffer_rate = pi - po;
    }
}

MADRONA_NO_INLINE bool allocOnePortDefault(
    Sim &sim,
    Context &ctx,
    int32_t port_id,
    double port_bw,
    double live_sum_in,
    PortBuffer &port_buf,
    PortPfcState *pfc_state,
    Entity *live_tags,
    int32_t num_live,
    PortCachedHints &hints,
    double &out_total)
{
    PriorityBuffer &pb = port_buf.prior_bufs[0];
    bool has_buffer = pb.buf_cnt > 1e-15 && pb.num_chunks > 0;
    if (!has_buffer && live_sum_in < 1e-18) {
        return false;
    }
    bool is_congested = live_sum_in >= port_bw;
    if (has_buffer) {
        BufferChunk &oldest = pb.buf_chunks[pb.head];
        if (!assignChunkWeightedOutBW(sim, ctx, port_id, oldest, port_bw) &&
            live_sum_in > 1e-15) {
            for (int32_t i = 0; i < num_live; i++) {
                FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
                tag.out_bw = port_bw * (tag.in_bw / live_sum_in);
            }
        }
    } else if (live_sum_in >= 1e-18) {
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
    if (pfc_state) {
        for (int32_t i = 0; i < num_live; i++) {
            FlowTagState &tag = ctx.get<FlowTagState>(live_tags[i]);
            if (pfc_state->paused[std::clamp(tag.priority, 0,
                    PFC_MAX_PRIORITY - 1)] != 0) {
                tag.out_bw = 0.0;
            }
        }
    }
    for (int32_t i = 0; i < num_live; i++) {
        out_total += ctx.get<FlowTagState>(live_tags[i]).out_bw;
    }
    if (sim.enableBuffer && has_buffer) {
        BufferChunk &front = pb.buf_chunks[pb.head];
        if (port_bw > 1e-6 && front.chunk_bytes > 1e-15) {
            Time td = front.chunk_bytes / port_bw;
            if (td > 1e-15) {
                if (!hints.has_drain_hint || td < hints.drain_hint_t) {
                    hints.has_drain_hint = 1;
                    hints.drain_hint_t = td;
                }
            }
        }
    }
    pb.net_buffer_rate = live_sum_in - out_total;
    return true;
}

} // namespace

void Sim::allocOnePort(
    Context &ctx,
    int32_t port_id,
    PortState &port_state,
    PortBuffer &port_buf_ref,
    DirtyPort &dirty,
    PortPfcConfig &,
    PortPfcState &pfc_state_ref,
    PortCachedHints &hints,
    PortDrainHint &drain_hint,
    PortCleanup &cleanup,
    PortTraceLast &trace,
    PortTagList &tag_list)
{
    resetAllocPhaseScratch(hints, drain_hint, cleanup, trace);

    if (dirty.isDirty == 0) {
        return;
    }
    trace.was_dirty_at_alloc = 1;
    // Legacy code called clearBacklogDrainTimer(pid) up-front for every
    // dirty port; we defer the clear to flushPortDrainHints so each port
    // only touches its own hint buffer in the parallel phase.
    drain_hint.want_clear = 1;

    double port_bw = port_state.port_bw;
    PortBuffer *port_buf = &port_buf_ref;
    PortPfcState *pfc_state = enablePfc ? &pfc_state_ref : nullptr;

    materializeAllocPortBuffer(*this, ctx, *port_buf, tag_list);

    AllocTagScratch scratch {};
    collectAllocTagScratch(*this, ctx, tag_list, cleanup, scratch);
    if (scratch.num_tags == 0) {
        return;
    }
    trace.alloc_num_tags = scratch.num_tags;

    if (scratch.num_live == 0) {
        return;
    }
    trace.alloc_num_live = scratch.num_live;

    bool is_dest_only = detectDestOnlyAllocPort(
        *this, ctx, port_id, scratch.live_tags, scratch.num_live);

    double out_total = 0.0;
    bool have_alloc_result = true;
    if ((qosMode == QOS_SP || qosMode == QOS_WRR) && !is_dest_only) {
        if (qosMode == QOS_SP) {
            allocOnePortQoSSP(*this, ctx, port_id, port_bw, *port_buf,
                pfc_state, scratch.live_tags, scratch.num_live, hints,
                out_total);
        } else {
            allocOnePortQoSWRR(*this, ctx, port_id, port_bw, *port_buf,
                pfc_state, scratch.live_tags, scratch.num_live, hints,
                out_total);
        }
    } else if (is_dest_only && qosMode != QOS_NONE) {
        allocOnePortDestOnly(ctx, port_bw, *port_buf, scratch.live_tags,
            scratch.num_live, out_total);
    } else {
        have_alloc_result = allocOnePortDefault(*this, ctx, port_id,
            port_bw, scratch.live_sum_in, *port_buf, pfc_state,
            scratch.live_tags, scratch.num_live, hints, out_total);
    }

    if (!have_alloc_result) {
        return;
    }

    finalizeAllocPortState(*this, ctx, *port_buf, hints, drain_hint, trace,
        scratch.live_tags, scratch.num_live, port_bw, scratch.live_sum_in,
        out_total, is_dest_only);
}

// Phase C: per-Port downstream emit worker. Each port only sees its own
// DirtyPort / PortOutbox / PortTraceLast and the tags whose tagIndex entry
// belongs to this port (read-only scan of tagIndex). DelayedEvents are
// written into the port's own outbox; the flushPortOutbox singleton
// appends them to Sim::delayedEvents in port_id ascending order so the
// deliverEvents order matches the legacy sequential implementation.
void Sim::emitOnePort(
    Context &ctx,
    int32_t port_id,
    PortState & /*port_state*/,
    DirtyPort &dirty,
    PortOutbox &outbox,
    PortTraceLast &trace,
    PortTagList &tag_list)
{
    outbox.num_events = 0;
    trace.emit_is_dirty = 0;
    trace.emit_arrival_count = 0;
    trace.emit_bwupdate_count = 0;

    if (dirty.isDirty == 0) {
        return;
    }
    trace.emit_is_dirty = 1;

    const bool log_enabled =
        compiledSystemLogEnabled("emit_tag", systemLogStep);

    // Phase D: iterate PortTagList instead of scanning the global tagIndex.
    for (int32_t i = 0; i < tag_list.count; i++) {
        Entity tag_e = tag_list.tags[i];
        if (tag_e == Entity::none()) {
            continue;
        }
        FlowTagState &tag = ctx.get<FlowTagState>(tag_e);
        if (tag.next_port_id < 0) {
            continue;
        }
        if (tag.downstream_created == 0) {
            if (tag.out_bw > 1e-15) {
                if (outbox.num_events < MAX_PORT_OUTBOX) {
                    DelayedEvent &ev = outbox.events[outbox.num_events++];
                    ev = DelayedEvent {};
                    ev.t = computePropagationTimeForPort(
                        tag.port_id, tag.next_port_id);
                    ev.type = DelayedEvent::Type::Arrival;
                    ev.arrival = FlowArrivalEv {
                        .port_id = tag.next_port_id,
                        .flow_id = tag.flow_id,
                        .size = 0.0,
                        .in_bw = tag.out_bw,
                        .is_source = 0,
                        .priority = tag.priority,
                    };
                    tag.downstream_created = 1;
                    trace.emit_arrival_count += 1;
                }
            }
            continue;
        }
        if (tag.out_bw == tag.prev_out_bw) {
            continue;
        }
        if (outbox.num_events < MAX_PORT_OUTBOX) {
            DelayedEvent &ev = outbox.events[outbox.num_events++];
            ev = DelayedEvent {};
            ev.t = computePropagationTimeForPort(
                tag.port_id, tag.next_port_id);
            ev.type = DelayedEvent::Type::BwUpdate;
            ev.bwupd = BwUpdateEv {
                .port_id = tag.next_port_id,
                .flow_id = tag.flow_id,
                .in_bw = tag.out_bw,
            };
            trace.emit_bwupdate_count += 1;
            if (log_enabled) {
                printSystemEmitBwUpdateTag(
                    systemLogStep, now, port_id,
                    tag.flow_id, tag.out_bw, tag.prev_out_bw);
            }
        }
    }
}

}
