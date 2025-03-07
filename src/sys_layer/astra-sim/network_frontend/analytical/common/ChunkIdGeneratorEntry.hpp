/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/
#define CUDA_HPP
#pragma once



#include <cassert>

namespace AstraSimAnalytical {

/**
 * ChunkIdGeneratorEntry tracks the chunk id generated
 * for sim_send() and sim_recv() calls
 * per each (tag, src, dest, chunk_size) tuple.
 */
class ChunkIdGeneratorEntry {
  public:
    /**
     * Constructor.
     */
    ChunkIdGeneratorEntry() noexcept;

    /**
     * Get send id.
     *
     * @return send id
     */
    [[nodiscard]] int get_send_id() const noexcept;

    /**
     * Get recv id.
     *
     * @return recv id
     */
    [[nodiscard]] int get_recv_id() const noexcept;

    /**
     * Increment send id.
     */
    void increment_send_id() noexcept;

    /**
     * Increment recv id.
     */
    void increment_recv_id() noexcept;

  private:
    /// send id
    int send_id;
    /// recv id
    int recv_id;
};

}  // namespace AstraSimAnalytical
