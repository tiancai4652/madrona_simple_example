#pragma once

#include <cassert>

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 回调函数类型定义
using CallbackFunc = void (*)(void*);

// 回调事件类
class CallbackEvent {
private:
    CallbackFunc callback_;
    void* arg_;
    bool registered_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    CallbackEvent() : callback_(nullptr), arg_(nullptr), registered_(false) {}

    // 注册回调
    CUDA_HOST_DEVICE
    void register_callback(CallbackFunc callback, void* arg) {
        assert(!registered_);
        callback_ = callback;
        arg_ = arg;
        registered_ = true;
    }

    // 检查是否已注册
    CUDA_HOST_DEVICE
    bool is_registered() const {
        return registered_;
    }

    // 触发回调
    CUDA_HOST_DEVICE
    void invoke() const {
        assert(registered_);
        callback_(arg_);
    }

    // 清除回调
    CUDA_HOST_DEVICE
    void clear() {
        callback_ = nullptr;
        arg_ = nullptr;
        registered_ = false;
    }
};

// 回调跟踪器条目
template<typename Key>
class CallbackTrackerEntry {
private:
    CallbackEvent send_event_;
    CallbackEvent recv_event_;
    bool transmission_finished_;
    Key key_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    CallbackTrackerEntry() : transmission_finished_(false) {}

    CUDA_HOST_DEVICE
    explicit CallbackTrackerEntry(const Key& key) 
        : transmission_finished_(false), key_(key) {}

    // 注册发送回调
    CUDA_HOST_DEVICE
    void register_send_callback(CallbackFunc callback, void* arg) {
        send_event_.register_callback(callback, arg);
    }

    // 注册接收回调
    CUDA_HOST_DEVICE
    void register_recv_callback(CallbackFunc callback, void* arg) {
        recv_event_.register_callback(callback, arg);
    }

    // 检查传输是否完成
    CUDA_HOST_DEVICE
    bool is_transmission_finished() const {
        return transmission_finished_;
    }

    // 设置传输完成
    CUDA_HOST_DEVICE
    void set_transmission_finished() {
        transmission_finished_ = true;
    }

    // 检查是否两个回调都已注册
    CUDA_HOST_DEVICE
    bool both_callbacks_registered() const {
        return send_event_.is_registered() && recv_event_.is_registered();
    }

    // 触发发送处理器
    CUDA_HOST_DEVICE
    void invoke_send_handler() {
        send_event_.invoke();
    }

    // 触发接收处理器
    CUDA_HOST_DEVICE
    void invoke_recv_handler() {
        recv_event_.invoke();
    }

    // 获取键值
    CUDA_HOST_DEVICE
    const Key& get_key() const {
        return key_;
    }
};

// 回调跟踪器
template<typename Key, size_t MaxEntries>
class CallbackTracker {
private:
    FixedMap<Key, CallbackTrackerEntry<Key>, MaxEntries> tracker_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    CallbackTracker() {}

    // 搜索条目
    CUDA_HOST_DEVICE
    CallbackTrackerEntry<Key>* search_entry(const Key& key) {
        return tracker_.find(key);
    }

    // 创建新条目
    CUDA_HOST_DEVICE
    CallbackTrackerEntry<Key>* create_new_entry(const Key& key) {
        bool success = tracker_.insert(key, CallbackTrackerEntry<Key>(key));
        assert(success);
        return tracker_.find(key);
    }

    // 删除条目
    CUDA_HOST_DEVICE
    void pop_entry(const Key& key) {
        bool success = tracker_.erase(key);
        assert(success);
    }
};

} // namespace custom 