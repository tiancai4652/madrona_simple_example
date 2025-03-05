/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/ChunkIdGenerator.hh"
#include <cassert>
#include "sys_layer/containers/FixedMap.hpp"
#include "sys_layer/containers/FixedTuple.hpp"

using namespace AstraSimAnalytical;

CUDA_HOST_DEVICE
ChunkIdGenerator::ChunkIdGenerator() noexcept {
    // 初始化映射
    chunk_id_map.clear();
}

CUDA_HOST_DEVICE
int ChunkIdGenerator::create_send_chunk_id(
    const int tag,
    const int src,
    const int dest,
    const ChunkSize chunk_size) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);

    // 创建键
    const auto key = make_fixed_tuple(tag, src, dest, chunk_size);

    // 查找或创建条目
    ChunkIdGeneratorEntry* entry = chunk_id_map.find(key);
    if (!entry) {
        bool success = chunk_id_map.insert(key, ChunkIdGeneratorEntry());
        assert(success);
        entry = chunk_id_map.find(key);
    }

    // 增加并返回ID
    entry->increment_send_id();
    return entry->get_send_id();
}

CUDA_HOST_DEVICE
int ChunkIdGenerator::create_recv_chunk_id(
    const int tag,
    const int src,
    const int dest,
    const ChunkSize chunk_size) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);

    // 创建键
    const auto key = make_fixed_tuple(tag, src, dest, chunk_size);

    // 查找或创建条目
    ChunkIdGeneratorEntry* entry = chunk_id_map.find(key);
    if (!entry) {
        bool success = chunk_id_map.insert(key, ChunkIdGeneratorEntry());
        assert(success);
        entry = chunk_id_map.find(key);
    }

    // 增加并返回ID
    entry->increment_recv_id();
    return entry->get_recv_id();
}
