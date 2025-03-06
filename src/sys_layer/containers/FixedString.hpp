#pragma once

#include <cstring>

namespace custom {

// 固定大小的字符串类
template<size_t MaxSize>
class FixedString {
private:
    char data_[MaxSize];
    size_t length_;

public:
    // 构造函数
    
    FixedString() : length_(0) {
        data_[0] = '\0';
    }

    
    FixedString(const char* str) {
        assign(str);
    }

    // 赋值操作
    
    void assign(const char* str) {
        length_ = 0;
        while (str[length_] != '\0' && length_ < MaxSize - 1) {
            data_[length_] = str[length_];
            ++length_;
        }
        data_[length_] = '\0';
    }

    // 访问操作
    
    const char* c_str() const { return data_; }

    
    size_t length() const { return length_; }

    
    bool empty() const { return length_ == 0; }

    // 比较操作
    
    bool operator==(const FixedString& other) const {
        if (length_ != other.length_) {
            return false;
        }
        for (size_t i = 0; i < length_; i++) {
            if (data_[i] != other.data_[i]) {
                return false;
            }
        }
        return true;
    }

    
    bool operator==(const char* str) const {
        size_t i = 0;
        while (i < length_ && str[i] != '\0') {
            if (data_[i] != str[i]) {
                return false;
            }
            i++;
        }
        return i == length_ && str[i] == '\0';
    }
};

} // namespace custom 