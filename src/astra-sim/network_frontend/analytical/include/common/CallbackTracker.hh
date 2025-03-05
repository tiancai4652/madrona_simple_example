/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CallbackTrackerEntry.hh"
#include "common/ChunkIdGenerator.hh"
#include "sys_layer/containers/FixedMap.hpp"
#include "sys_layer/containers/FixedTuple.hpp"

namespace AstraSimAnalytical {

/**
 * CallbackTracker跟踪由(tag, src, dest, chunk_size, chunk_id)元组标识的
 * 每个块的sim_send()和sim_recv()回调。
 */
template<size_t MaxEntries>
class CallbackTracker {
  public:
    /// Key = (tag, src, dest, chunk_size, chunk_id)
    using Key = custom::FixedTuple<int, int, int, ChunkSize, int>;

    CUDA_HOST_DEVICE
    CallbackTracker() noexcept {}

    /**
     * 搜索由(tag, src, dest, chunk_size, chunk_id)元组标识的条目
     */
    CUDA_HOST_DEVICE
    custom::FixedOptional<CallbackTrackerEntry*> search_entry(
        int tag, int src, int dest, ChunkSize chunk_size, int chunk_id) noexcept {
        Key key(tag, src, dest, chunk_size, chunk_id);
        auto* entry = tracker.find(key);
        if (entry) {
            return custom::FixedOptional<CallbackTrackerEntry*>(entry);
        }
        return custom::FixedOptional<CallbackTrackerEntry*>();
    }

    /**
     * 创建由(tag, src, dest, chunk_size, chunk_id)元组标识的新条目
     */
    CUDA_HOST_DEVICE
    CallbackTrackerEntry* create_new_entry(
        int tag, int src, int dest, ChunkSize chunk_size, int chunk_id) noexcept {
        Key key(tag, src, dest, chunk_size, chunk_id);
        auto [it, inserted] = tracker.insert(key, CallbackTrackerEntry());
        return &(it->value);
    }

    /**
     * 移除由(tag, src, dest, chunk_size, chunk_id)元组标识的条目
     */
    CUDA_HOST_DEVICE
    void pop_entry(int tag, int src, int dest, ChunkSize chunk_size, int chunk_id) noexcept {
        Key key(tag, src, dest, chunk_size, chunk_id);
        tracker.erase(key);
    }

  private:
    /// 从(tag, src, dest, chunk_size, chunk_id)元组到CallbackTrackerEntry的映射
    custom::FixedMap<Key, CallbackTrackerEntry, MaxEntries> tracker;
};

}  // namespace AstraSimAnalytical
