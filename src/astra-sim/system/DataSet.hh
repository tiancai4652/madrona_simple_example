/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/StreamStat.hh"
#include "sys_layer/containers/FixedTuple.hpp"

namespace AstraSim {

/**
 * 数据集类，用于管理流处理
 */
class DataSet : public Callable, public StreamStat {
public:
    CUDA_HOST_DEVICE DataSet(int total_streams);
    CUDA_HOST_DEVICE void set_notifier(Callable* layer, EventType event);
    CUDA_HOST_DEVICE void notify_stream_finished(StreamStat* data);
    CUDA_HOST_DEVICE void call(EventType event, CallData* data) override;
    CUDA_HOST_DEVICE bool is_finished();

    static int id_auto_increment;  ///< 自增ID
    int my_id;                    ///< 当前数据集ID
    int total_streams;            ///< 总流数
    int finished_streams;         ///< 已完成流数
    bool finished;                ///< 是否完成
    bool active;                  ///< 是否活跃
    Tick finish_tick;            ///< 完成时间戳
    Tick creation_tick;          ///< 创建时间戳
    custom::FixedTuple<Callable*, EventType>* notifier;  ///< 通知器
};

}  // namespace AstraSim
