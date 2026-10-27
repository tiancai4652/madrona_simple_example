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
#include <ctime>
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

// 高水位探针：记录某量的历史最大值（018-layer0-GPUfit 收常量用）。
// 同一 env 开关；退出时随汇总打印；GPU 下展开为空。
namespace hwm {
constexpr uint32_t kMaxGauges = 32;
struct Gauge {
    const char *name = nullptr;
    uint64_t max = 0;
};
inline Gauge gauges[kMaxGauges];
inline uint32_t num_gauges = 0;
inline void dump();
inline void record(const char *name, uint64_t v)
{
    impl::ensureInit();
    if (!impl::enabled) {
        return;
    }
    if (num_gauges == 0) {
        std::atexit(dump);
    }
    for (uint32_t i = 0; i < num_gauges; i++) {
        if (gauges[i].name == name) {
            if (v > gauges[i].max) {
                gauges[i].max = v;
            }
            return;
        }
    }
    if (num_gauges >= kMaxGauges) {
        return;
    }
    gauges[num_gauges].name = name;
    gauges[num_gauges].max = v;
    num_gauges += 1;
}
inline void dump()
{
    if (!impl::enabled) {
        return;
    }
    for (uint32_t i = 0; i < num_gauges; i++) {
        std::fprintf(stderr, "[PROBE-HWM] %s max=%llu\n",
                     gauges[i].name,
                     (unsigned long long)gauges[i].max);
    }
    std::fflush(stderr);
}
} // namespace hwm

// 计时探针：SIM_PROBE_TIMED("名字") 放在系统函数入口，统计该系统
// 累计耗时/调用次数（018 索引优化找热点用）。同一 env 开关；GPU 下展开为空。
namespace timed {
constexpr uint32_t kMaxTimers = 128;
struct Entry {
    const char *name = nullptr;
    uint64_t calls = 0;
    uint64_t total_ns = 0;
};
inline Entry entries[kMaxTimers];
inline uint32_t num_entries = 0;
inline uint64_t nowNs()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
inline void dump()
{
    if (!impl::enabled) {
        return;
    }
    for (uint32_t i = 0; i < num_entries; i++) {
        double total_ms = (double)entries[i].total_ns / 1000000.0;
        double mean_us = entries[i].calls ?
            (double)entries[i].total_ns / (double)entries[i].calls / 1000.0 : 0.0;
        std::fprintf(stderr, "[PROBE-TIME] sys=%s calls=%llu total_ms=%.1f mean_us=%.2f\n",
                     entries[i].name, (unsigned long long)entries[i].calls,
                     total_ms, mean_us);
    }
    std::fflush(stderr);
}
struct Scope {
    int idx = -1;
    uint64_t t0 = 0;
    Scope(const char *name)
    {
        impl::ensureInit();
        if (!impl::enabled) {
            return;
        }
        t0 = nowNs();
        for (uint32_t i = 0; i < num_entries; i++) {
            if (entries[i].name == name) {
                idx = (int)i;
                return;
            }
        }
        if (num_entries < kMaxTimers) {
            idx = (int)num_entries++;
            entries[idx].name = name;
            std::atexit(dump);
        }
    }
    ~Scope()
    {
        if (idx < 0) {
            return;
        }
        entries[idx].calls += 1;
        entries[idx].total_ns += nowNs() - t0;
    }
};
} // namespace timed

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
// 顺带启动作用域计时（同名，如一函数内调两次会编译报错，届时改名）。
#define SIM_PROBE_RECORD(name)                                    \
    ::madsimple::probe::timed::Scope sim_probe_scope_obj(name);    \
    do {                                                          \
        static uint64_t *sim_probe_ctr =                                          \
            ::madsimple::probe::slot(name);                       \
        if (sim_probe_ctr != nullptr) {                           \
            ++(*sim_probe_ctr);                                   \
        }                                                         \
    } while (0)

#define SIM_PROBE_FRAME_BEGIN() ::madsimple::probe::beginFrame()
#define SIM_PROBE_FRAME_END() ::madsimple::probe::endFrame()

// 高水位：SIM_PROBE_HWM("名字", 值)；计时：SIM_PROBE_TIMED("名字") 放函数入口。
#define SIM_PROBE_HWM(name, v) ::madsimple::probe::hwm::record(name, \
    (uint64_t)(v))
#define SIM_PROBE_TIMED(name) ::madsimple::probe::timed::Scope \
    sim_probe_timed_obj_##__LINE__(name)

#else // MADRONA_GPU_MODE

#define SIM_PROBE_RECORD(name) ((void)0)
#define SIM_PROBE_FRAME_BEGIN() ((void)0)
#define SIM_PROBE_FRAME_END() ((void)0)
#define SIM_PROBE_HWM(name, v) ((void)0)
#define SIM_PROBE_TIMED(name) ((void)0)

#endif // MADRONA_GPU_MODE
