#pragma once

// =============================================================================
// 并行度探针（期1 · env-gated 临时开关）
//
// 用法：环境变量 MADRONA_PARALLELISM_PROBE=1 启用；未设置/为 0 时零开销。
// 口径：每个任务图系统在自己的每个实例入口调用 SIM_PROBE_RECORD("名字")，
//       探针按帧聚合出"该系统本帧实际执行的实例数"，运行结束打印每系统的
//       P50/P90/max/mean 汇总到 stderr。
//
// 边界说明（重要）：
// * 本探针是纯测量工具，输出只走 stderr，绝不进入三本账
//   （FCT CSV / 推理 stats CSV / summary json）。
// * 仅 CPU 构建生效：所有宏在 MADRONA_GPU_MODE 下展开为空，GPU 编译零改动。
// * 计数器为全进程一份：单 world 运行时口径精确；多 world 并发时各线程
//   会累加到同一槽位（本期所有验收场景都是 num_worlds=1）。
// * 未在帧内执行的实例（如某系统查询命中 0 个实体）不会注册调用点，
//   汇总里的 frames = 该系统出现过实例的帧数。
// =============================================================================

#ifndef MADRONA_GPU_MODE

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace madsimple::probe {

namespace impl {

constexpr uint32_t kMaxSystems = 128;

struct SystemSeries {
    const char *name = nullptr;
    std::vector<uint64_t> per_frame; // 每帧实例数（按帧序）
    uint64_t cur = 0;                // 当前帧累计
};

inline SystemSeries series[kMaxSystems];
inline uint32_t num_series = 0;
inline uint64_t frames_seen = 0;
inline bool enabled = false;
inline bool inited = false;

inline void dumpSummary()
{
    if (!enabled || num_series == 0) {
        return;
    }
    std::fprintf(stderr,
        "[PROBE-SUMMARY] frames=%llu systems=%u\n",
        (unsigned long long)frames_seen, (unsigned)num_series);
    for (uint32_t i = 0; i < num_series; i++) {
        SystemSeries &s = series[i];
        if (s.per_frame.empty()) {
            continue;
        }
        std::vector<uint64_t> sorted = s.per_frame;
        std::sort(sorted.begin(), sorted.end());
        auto pct = [&](double p) -> uint64_t {
            // 线性插值分位数（与 numpy.percentile 的 linear 插值一致）
            double idx = p * (double)(sorted.size() - 1);
            size_t lo = (size_t)idx;
            size_t hi = (lo + 1 < sorted.size()) ? lo + 1 : lo;
            double frac = idx - (double)lo;
            return (uint64_t)(0.5 + (double)sorted[lo] * (1.0 - frac) +
                (double)sorted[hi] * frac);
        };
        double mean = 0.0;
        for (uint64_t v : s.per_frame) {
            mean += (double)v;
        }
        mean /= (double)s.per_frame.size();
        std::fprintf(stderr,
            "[PROBE] sys=%s frames=%zu p50=%llu p90=%llu max=%llu "
            "mean=%.2f\n",
            s.name, s.per_frame.size(),
            (unsigned long long)pct(0.50),
            (unsigned long long)pct(0.90),
            (unsigned long long)sorted.back(), mean);
    }
    std::fflush(stderr);
}

inline void ensureInit()
{
    if (inited) {
        return;
    }
    inited = true;
    const char *env = std::getenv("MADRONA_PARALLELISM_PROBE");
    enabled = env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    if (enabled) {
        std::atexit(dumpSummary);
    }
}

} // namespace impl

// 返回该调用点的计数槽；未启用时返回 nullptr（调用点缓存住，零开销）。
inline uint64_t *slot(const char *name)
{
    impl::ensureInit();
    if (!impl::enabled) {
        return nullptr;
    }
    for (uint32_t i = 0; i < impl::num_series; i++) {
        if (impl::series[i].name == name) {
            return &impl::series[i].cur;
        }
    }
    if (impl::num_series >= impl::kMaxSystems) {
        return nullptr;
    }
    impl::SystemSeries &s = impl::series[impl::num_series++];
    s.name = name;
    s.cur = 0;
    return &s.cur;
}

// 帧首（beginScheduleStepSystem）：新帧计数清零。
inline void beginFrame()
{
    impl::ensureInit();
    if (!impl::enabled) {
        return;
    }
    for (uint32_t i = 0; i < impl::num_series; i++) {
        impl::series[i].cur = 0;
    }
}

// 帧尾（updateSystemStatusStepSystem）：把本帧各系统实例数存档。
inline void endFrame()
{
    impl::ensureInit();
    if (!impl::enabled) {
        return;
    }
    for (uint32_t i = 0; i < impl::num_series; i++) {
        impl::series[i].per_frame.push_back(impl::series[i].cur);
    }
    impl::frames_seen += 1;
    // 长跑防丢：每 500 帧落一次运行中汇总（最终汇总仍以 atexit 为准）
    if (impl::frames_seen % 500 == 0) {
        impl::dumpSummary();
    }
}

} // namespace madsimple::probe

// 每个系统实例入口调用一次；static 指针缓存使稳态开销为一次判空。
#define SIM_PROBE_RECORD(name)                                    \
    do {                                                          \
        static uint64_t *sim_probe_ctr =                                          \
            ::madsimple::probe::slot(name);                       \
        if (sim_probe_ctr != nullptr) {                           \
            ++(*sim_probe_ctr);                                   \
        }                                                         \
    } while (0)

#define SIM_PROBE_FRAME_BEGIN() ::madsimple::probe::beginFrame()
#define SIM_PROBE_FRAME_END() ::madsimple::probe::endFrame()

#else // MADRONA_GPU_MODE

#define SIM_PROBE_RECORD(name) ((void)0)
#define SIM_PROBE_FRAME_BEGIN() ((void)0)
#define SIM_PROBE_FRAME_END() ((void)0)

#endif // MADRONA_GPU_MODE
