#pragma once

#include <cassert>
#include "FixedMap.hpp"
#include "FixedTuple.hpp"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 块ID生成器条目
class ChunkIdGeneratorEntry {
private:
    int send_id_;
    int recv_id_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    ChunkIdGeneratorEntry() : send_id_(-1), recv_id_(-1) {}

    // 获取发送ID
    CUDA_HOST_DEVICE
    int get_send_id() const {
        assert(send_id_ >= 0);
        return send_id_;
    }

    // 获取接收ID
    CUDA_HOST_DEVICE
    int get_recv_id() const {
        assert(recv_id_ >= 0);
        return recv_id_;
    }

    // 增加发送ID
    CUDA_HOST_DEVICE
    void increment_send_id() {
        send_id_++;
    }

    // 增加接收ID
    CUDA_HOST_DEVICE
    void increment_recv_id() {
        recv_id_++;
    }
};

// 块ID生成器
template<size_t MaxEntries>
class ChunkIdGenerator {
private:
    using ChunkKey = FixedTuple<int, int, int, uint64_t>; // tag, src, dest, chunk_size
    FixedMap<ChunkKey, ChunkIdGeneratorEntry, MaxEntries> chunk_id_map_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    ChunkIdGenerator() {}

    // 创建发送块ID
    CUDA_HOST_DEVICE
    int create_send_chunk_id(
        int tag,
        int src,
        int dest,
        uint64_t chunk_size) {
        assert(tag >= 0);
        assert(src >= 0);
        assert(dest >= 0);
        assert(chunk_size > 0);

        // 创建键
        auto key = make_fixed_tuple(tag, src, dest, chunk_size);

        // 查找或创建条目
        ChunkIdGeneratorEntry* entry = chunk_id_map_.find(key);
        if (!entry) {
            bool success = chunk_id_map_.insert(key, ChunkIdGeneratorEntry());
            assert(success);
            entry = chunk_id_map_.find(key);
        }

        // 增加并返回ID
        entry->increment_send_id();
        return entry->get_send_id();
    }

    // 创建接收块ID
    CUDA_HOST_DEVICE
    int create_recv_chunk_id(
        int tag,
        int src,
        int dest,
        uint64_t chunk_size) {
        assert(tag >= 0);
        assert(src >= 0);
        assert(dest >= 0);
        assert(chunk_size > 0);

        // 创建键
        auto key = make_fixed_tuple(tag, src, dest, chunk_size);

        // 查找或创建条目
        ChunkIdGeneratorEntry* entry = chunk_id_map_.find(key);
        if (!entry) {
            bool success = chunk_id_map_.insert(key, ChunkIdGeneratorEntry());
            assert(success);
            entry = chunk_id_map_.find(key);
        }

        // 增加并返回ID
        entry->increment_recv_id();
        return entry->get_recv_id();
    }

    // 清空生成器
    CUDA_HOST_DEVICE
    void clear() {
        chunk_id_map_.clear();
    }
};

} // namespace custom 