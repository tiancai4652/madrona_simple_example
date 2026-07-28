#include "fake_system.hpp"

namespace madsimple {

namespace {

bool containsExpectedFlow(const FakeSystemStats &fake,
                          const SysFlow *finished,
                          uint32_t count)
{
    uint32_t expected_flow_id = fake.flow_id_base + fake.round;
    for (uint32_t i = 0; i < count; i++) {
        if (finished[i].flow_id == expected_flow_id) {
            return true;
        }
    }
    return false;

}

}

void fakeSystemStep(Engine &ctx, FakeSystemStats &fake)
{
    using namespace net_sys_interface;

    if (fake.enabled == 0) {
        fake.state = FakeSystemState::Disabled;
        return;
    }

    switch (fake.state) {
    case FakeSystemState::Disabled:
        fake.state = FakeSystemState::Init;
        break;
    case FakeSystemState::Init: {
        uint64_t now_ns = getCurrentTime(ctx);
        fake.compute_finish_ns = now_ns + fake.compute_duration_ns;
        if (!addEvent(ctx, fake.compute_finish_ns)) {
            fake.state = FakeSystemState::Failed;
            fake.error_code = 1;
            return;
        }
        fake.events_added += 1;
        fake.state = FakeSystemState::Computing;
        break;
    }
    case FakeSystemState::Computing:
        if (getCurrentTime(ctx) >= fake.compute_finish_ns) {
            uint32_t flow_id = fake.flow_id_base + fake.round;
            setFlow(ctx, fake.npu_id, fake.src_npu, fake.dst_npu,
                    fake.flow_size, flow_id);
            fake.submitted_flows += 1;
            fake.last_submit_time_ns = getCurrentTime(ctx);
            fake.state = FakeSystemState::WaitingFlow;
        }
        break;
    case FakeSystemState::WaitingFlow: {
        SysFlow finished[MAX_FAKE_FINISHED_FLOWS] {};
        uint32_t count = checkFlowFinish(ctx, fake.npu_id, finished,
                                         MAX_FAKE_FINISHED_FLOWS);
        bool found = containsExpectedFlow(fake, finished, count);
        // This NPU is the only consumer of its own finished-flow mailbox,
        // so it is safe to clear immediately after reading in the same
        // per-NPU step (see NpuFlowFinishedList doc comment in types.hpp;
        // a future send/recv dual-consumer NPU would need to coordinate
        // clearing between the two instead of clearing unconditionally
        // here).
        if (count > 0) {
            clearFlowFinishQueue(ctx, fake.npu_id);
        }
        if (found) {
            fake.completed_flows += 1;
            fake.last_complete_time_ns = getCurrentTime(ctx);
            fake.round += 1;
            fake.state = fake.round >= fake.max_rounds ?
                FakeSystemState::Finished : FakeSystemState::Init;
        }
        break;
    }
    case FakeSystemState::Finished:
    case FakeSystemState::Failed:
        break;
    }
}

}
