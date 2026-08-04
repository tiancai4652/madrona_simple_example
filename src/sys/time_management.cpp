#include "time_management.hpp"
#include "net_sys_interface.hpp"
#include "sys_config.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

using madsimple::net_sys_interface::addSimtime;
using madsimple::net_sys_interface::addEvent;
using madsimple::net_sys_interface::getCurrentTime;
using madsimple::net_sys_interface::isExistedFlow;

// Log formatting macros
#define LOG_SUBSEPARATOR "-------------------------------------------------------------------------------"

#if SYS_LOG
#define LOG_SYS_HEADER(sys_name, sys_desc) \
    printf("\n[SYS-%s] %s\n%s\n", sys_name, sys_desc, LOG_SUBSEPARATOR);
#define LOG_TASK_START(task_name) \
    printf("  > %s\n", task_name);
#define LOG_INFO(msg, ...) \
    printf("  INFO: " msg "\n", ##__VA_ARGS__);
#else
#define LOG_SYS_HEADER(sys_name, sys_desc) \
    do                                     \
    {                                      \
    } while (0)
#define LOG_TASK_START(task_name) \
    do                            \
    {                             \
    } while (0)
#define LOG_INFO(msg, ...) \
    do                     \
    {                      \
    } while (0)
#endif

namespace madsimple::llm_system
{

    uint64_t usToNs(uint64_t microseconds)
    {
        return microseconds * 1000; // 1 us = 1,000 ns
    }


    void skipTime_add_time(Engine &ctx, uint64_t t_relative_ns)
    {
        uint64_t event_time_ns = getCurrentTime(ctx) + t_relative_ns;
        for (size_t i = 0; i < MAX_CHAKRA_NODES_PER_NPU; i++)
        {
            if (ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs[i] == 0)
            {
                ctx.get<NextProcessTimes>(ctx.data().next_process_time_entity).times_abs[i] = event_time_ns;
                addEvent(ctx, event_time_ns);
                break;
            }
        }
    }

    // Custom sorting function: bubble sort from large to small
    void customSortDescending(uint64_t *array, size_t length)
    {
        for (size_t i = 0; i < length - 1; ++i)
        {
            for (size_t j = 0; j < length - 1 - i; ++j)
            {
                if (array[j] < array[j + 1])
                { // Compare from large to small
                  // Swap two elements
                    uint64_t temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                }
            }
        }
    }

    uint64_t skipTime_sort_time_rMin(Engine &ctx)
    {
        uint64_t min_time = 0;
        for (size_t i = 0; i < MAX_CHAKRA_NODES_PER_NPU; ++i)
        {
            uint64_t candidate = ctx.get<NextProcessTimes>(
                ctx.data().next_process_time_entity).times_abs[i];
            if (candidate != 0 && (min_time == 0 || candidate < min_time)) {
                min_time = candidate;
            }
        }
        return min_time;
    }

    void skipTime_remove_time(Engine &ctx, uint64_t x_ns)
    {
        for (size_t i = 0; i < MAX_CHAKRA_NODES_PER_NPU; ++i)
        {
            uint64_t &candidate = ctx.get<NextProcessTimes>(
                ctx.data().next_process_time_entity).times_abs[i];
            if (candidate == x_ns) {
                candidate = 0;
                return;
            }
        }
    }

    uint16_t frame_skiptime = 0;
    void sys_checkSkipTime(Engine &ctx, NextProcessTimes &t)
    {
        (void)t;
        LOG_SYS_HEADER("4", "Time Skip Check System");
        LOG_TASK_START("Checking if simulation time skip is needed");

        frame_skiptime++;
        if (frame_skiptime / CHECK_SKIPTIME_INTERVAL_PER_FRAME == 0)
        {
            frame_skiptime = 0;
            if (!isExistedFlow(ctx))
            {
                uint64_t min_time = skipTime_sort_time_rMin(ctx);
                if (min_time != 0)
                {
                    addSimtime(ctx, min_time - getCurrentTime(ctx));
                    LOG_INFO("Time skipped to: %ld ns", getCurrentTime(ctx));
                    skipTime_remove_time(ctx, min_time);
                }
            }
        }
    }

}
