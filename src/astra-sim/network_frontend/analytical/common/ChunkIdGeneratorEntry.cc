/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/ChunkIdGeneratorEntry.hh"
#include <cassert>

using namespace AstraSimAnalytical;

CUDA_HOST_DEVICE
ChunkIdGeneratorEntry::ChunkIdGeneratorEntry() noexcept : send_id(-1), recv_id(-1) {}

CUDA_HOST_DEVICE
int ChunkIdGeneratorEntry::get_send_id() const noexcept {
    assert(send_id >= 0);

    return send_id;
}

CUDA_HOST_DEVICE
int ChunkIdGeneratorEntry::get_recv_id() const noexcept {
    assert(recv_id >= 0);

    return recv_id;
}

CUDA_HOST_DEVICE
void ChunkIdGeneratorEntry::increment_send_id() noexcept {
    send_id++;
}

CUDA_HOST_DEVICE
void ChunkIdGeneratorEntry::increment_recv_id() noexcept {
    recv_id++;
}
