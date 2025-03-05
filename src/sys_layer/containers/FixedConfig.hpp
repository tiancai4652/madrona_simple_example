#pragma once

#include <cassert>
#include "FixedString.hpp"
#include "FixedMap.hpp"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 配置选项类
template<typename T>
class ConfigOption {
private:
    T value_;
    T default_value_;
    bool has_value_;
    bool required_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    ConfigOption() : has_value_(false), required_(false) {}

    CUDA_HOST_DEVICE
    ConfigOption(const T& default_value, bool required = false)
        : value_(default_value), default_value_(default_value), 
          has_value_(false), required_(required) {}

    // 设置值
    CUDA_HOST_DEVICE
    void set_value(const T& value) {
        value_ = value;
        has_value_ = true;
    }

    // 获取值
    CUDA_HOST_DEVICE
    const T& get_value() const {
        return has_value_ ? value_ : default_value_;
    }

    // 检查是否已设置值
    CUDA_HOST_DEVICE
    bool has_value() const {
        return has_value_;
    }

    // 检查是否必需
    CUDA_HOST_DEVICE
    bool is_required() const {
        return required_;
    }

    // 重置为默认值
    CUDA_HOST_DEVICE
    void reset() {
        value_ = default_value_;
        has_value_ = false;
    }
};

// 配置解析器类
template<size_t MaxOptions>
class ConfigParser {
private:
    using OptionKey = FixedString<64>;
    FixedMap<OptionKey, ConfigOption<FixedString<256>>, MaxOptions> string_options_;
    FixedMap<OptionKey, ConfigOption<int>, MaxOptions> int_options_;
    FixedMap<OptionKey, ConfigOption<double>, MaxOptions> double_options_;
    FixedMap<OptionKey, ConfigOption<bool>, MaxOptions> bool_options_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    ConfigParser() {}

    // 添加字符串选项
    CUDA_HOST_DEVICE
    void add_string_option(const char* name, const char* default_value = "", bool required = false) {
        string_options_.insert(OptionKey(name), ConfigOption<FixedString<256>>(FixedString<256>(default_value), required));
    }

    // 添加整数选项
    CUDA_HOST_DEVICE
    void add_int_option(const char* name, int default_value = 0, bool required = false) {
        int_options_.insert(OptionKey(name), ConfigOption<int>(default_value, required));
    }

    // 添加浮点数选项
    CUDA_HOST_DEVICE
    void add_double_option(const char* name, double default_value = 0.0, bool required = false) {
        double_options_.insert(OptionKey(name), ConfigOption<double>(default_value, required));
    }

    // 添加布尔选项
    CUDA_HOST_DEVICE
    void add_bool_option(const char* name, bool default_value = false, bool required = false) {
        bool_options_.insert(OptionKey(name), ConfigOption<bool>(default_value, required));
    }

    // 设置选项值
    CUDA_HOST_DEVICE
    bool set_string_value(const char* name, const char* value) {
        auto* option = string_options_.find(OptionKey(name));
        if (option) {
            option->set_value(FixedString<256>(value));
            return true;
        }
        return false;
    }

    CUDA_HOST_DEVICE
    bool set_int_value(const char* name, int value) {
        auto* option = int_options_.find(OptionKey(name));
        if (option) {
            option->set_value(value);
            return true;
        }
        return false;
    }

    CUDA_HOST_DEVICE
    bool set_double_value(const char* name, double value) {
        auto* option = double_options_.find(OptionKey(name));
        if (option) {
            option->set_value(value);
            return true;
        }
        return false;
    }

    CUDA_HOST_DEVICE
    bool set_bool_value(const char* name, bool value) {
        auto* option = bool_options_.find(OptionKey(name));
        if (option) {
            option->set_value(value);
            return true;
        }
        return false;
    }

    // 获取选项值
    CUDA_HOST_DEVICE
    const FixedString<256>& get_string_value(const char* name) const {
        const auto* option = string_options_.find(OptionKey(name));
        assert(option != nullptr);
        return option->get_value();
    }

    CUDA_HOST_DEVICE
    int get_int_value(const char* name) const {
        const auto* option = int_options_.find(OptionKey(name));
        assert(option != nullptr);
        return option->get_value();
    }

    CUDA_HOST_DEVICE
    double get_double_value(const char* name) const {
        const auto* option = double_options_.find(OptionKey(name));
        assert(option != nullptr);
        return option->get_value();
    }

    CUDA_HOST_DEVICE
    bool get_bool_value(const char* name) const {
        const auto* option = bool_options_.find(OptionKey(name));
        assert(option != nullptr);
        return option->get_value();
    }

    // 验证所有必需选项都已设置
    CUDA_HOST_DEVICE
    bool validate() const {
        for (const auto& pair : string_options_) {
            if (pair.value.is_required() && !pair.value.has_value()) {
                return false;
            }
        }
        for (const auto& pair : int_options_) {
            if (pair.value.is_required() && !pair.value.has_value()) {
                return false;
            }
        }
        for (const auto& pair : double_options_) {
            if (pair.value.is_required() && !pair.value.has_value()) {
                return false;
            }
        }
        for (const auto& pair : bool_options_) {
            if (pair.value.is_required() && !pair.value.has_value()) {
                return false;
            }
        }
        return true;
    }
};

} // namespace custom 