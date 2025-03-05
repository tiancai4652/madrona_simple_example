/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CallbackTracker.hh"
#include <cassert>
#include "sys_layer/containers/FixedMap.hpp"
#include "sys_layer/containers/FixedTuple.hpp"
#include "sys_layer/containers/FixedOptional.hpp"

using namespace AstraSimAnalytical;

CUDA_HOST_DEVICE
CallbackTracker::CallbackTracker() noexcept {
    // 初始化跟踪器
    tracker.clear();
}

CUDA_HOST_DEVICE
FixedOptional<CallbackTrackerEntry*> CallbackTracker::search_entry(
    const int tag, const int src, const int dest, const ChunkSize chunk_size, const int chunk_id) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);
    assert(chunk_id >= 0);

    // 创建键并搜索条目
    const auto key = make_fixed_tuple(tag, src, dest, chunk_size, chunk_id);
    CallbackTrackerEntry* entry = tracker.find(key);

    // 条目不存在
    if (!entry) {
        return make_nullopt<CallbackTrackerEntry*>();
    }

    // 返回条目指针
    return make_optional(entry);
}

CUDA_HOST_DEVICE
CallbackTrackerEntry* CallbackTracker::create_new_entry(
    const int tag, const int src, const int dest, const ChunkSize chunk_size, const int chunk_id) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);
    assert(chunk_id >= 0);

    // 创建键
    const auto key = make_fixed_tuple(tag, src, dest, chunk_size, chunk_id);

    // 创建新的空条目
    bool success = tracker.insert(key, CallbackTrackerEntry());
    assert(success);

    // 返回条目指针
    return tracker.find(key);
}

CUDA_HOST_DEVICE
void CallbackTracker::pop_entry(
    const int tag, const int src, const int dest, const ChunkSize chunk_size, const int chunk_id) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);
    assert(chunk_id >= 0);

    // 创建键
    const auto key = make_fixed_tuple(tag, src, dest, chunk_size, chunk_id);

    // 查找条目
    CallbackTrackerEntry* entry = tracker.find(key);
    assert(entry != nullptr);  // 条目必须存在

    // 从跟踪器中删除条目
    bool success = tracker.erase(key);
    assert(success);
}
