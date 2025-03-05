/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSimAnalytical {

/**
 * ChunkIdGeneratorEntry跟踪为每个(tag, src, dest, chunk_size)元组
 * 生成的sim_send()和sim_recv()调用的块ID。
 */
class ChunkIdGeneratorEntry {
  public:
    /**
     * 构造函数
     */
    CUDA_HOST_DEVICE
    ChunkIdGeneratorEntry() noexcept;

    /**
     * 获取sim_send()调用的块ID
     */
    CUDA_HOST_DEVICE
    [[nodiscard]] int get_send_id() const noexcept;

    /**
     * 获取sim_recv()调用的块ID
     */
    CUDA_HOST_DEVICE
    [[nodiscard]] int get_recv_id() const noexcept;

    /**
     * 增加sim_send()调用的块ID
     */
    CUDA_HOST_DEVICE
    void increment_send_id() noexcept;

    /**
     * 增加sim_recv()调用的块ID
     */
    CUDA_HOST_DEVICE
    void increment_recv_id() noexcept;

  private:
    /// sim_send()调用的当前可用块ID
    int send_id;

    /// sim_recv()调用的当前可用块ID
    int recv_id;
};

}  // namespace AstraSimAnalytical
