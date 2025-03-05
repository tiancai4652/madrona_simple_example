/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/ChunkIdGeneratorEntry.hh"
#include <astra-network-analytical/common/Type.h>
#include "sys_layer/containers/FixedMap.hpp"
#include "sys_layer/containers/FixedTuple.hpp"

using namespace NetworkAnalytical;

namespace AstraSimAnalytical {

/**
 * ChunkIdGenerator为给定的(tag, src, dest, chunk_size)元组
 * 生成sim_send()和sim_recv()调用的唯一块ID。
 */
template<size_t MaxEntries>
class ChunkIdGenerator {
  public:
    /// Key = (tag, src, dest, chunk_size)
    using Key = custom::FixedTuple<int, int, int, ChunkSize>;

    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    ChunkIdGenerator() noexcept {}

    /**
     * 为sim_send()调用创建唯一块ID
     */
    CUDA_HOST_DEVICE
    [[nodiscard]] int create_send_chunk_id(int tag, int src, int dest, ChunkSize chunk_size) noexcept {
        Key key(tag, src, dest, chunk_size);
        auto [it, inserted] = chunk_id_map.insert(key, ChunkIdGeneratorEntry());
        auto* entry = &(it->value);
        const auto chunk_id = entry->get_send_id();
        entry->increment_send_id();
        return chunk_id;
    }

    /**
     * 为sim_recv()调用创建唯一块ID
     */
    CUDA_HOST_DEVICE
    [[nodiscard]] int create_recv_chunk_id(int tag, int src, int dest, ChunkSize chunk_size) noexcept {
        Key key(tag, src, dest, chunk_size);
        auto [it, inserted] = chunk_id_map.insert(key, ChunkIdGeneratorEntry());
        auto* entry = &(it->value);
        const auto chunk_id = entry->get_recv_id();
        entry->increment_recv_id();
        return chunk_id;
    }

  private:
    /// 从(tag, src, dest, chunk_size)元组到ChunkIdGeneratorEntry的映射
    custom::FixedMap<Key, ChunkIdGeneratorEntry, MaxEntries> chunk_id_map;
};

}  // namespace AstraSimAnalytical
