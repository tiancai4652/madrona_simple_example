#pragma once
#include <cstddef>

namespace custom {
template<typename Key, typename Value>
class CustomMap {
private:
    struct Node {
        Key key;
        Value value;
        bool occupied;
        Node() : occupied(false) {}
    };

    Node* nodes;
    size_t capacity;
    size_t size_;

    void resize(size_t newCapacity) {
        Node* newNodes = new Node[newCapacity];
        size_t oldCapacity = capacity;
        capacity = newCapacity;

        for (size_t i = 0; i < oldCapacity; ++i) {
            if (nodes[i].occupied) {
                insertNode(newNodes, newCapacity, nodes[i].key, nodes[i].value);
            }
        }

        delete[] nodes;
        nodes = newNodes;
    }

    void insertNode(Node* target, size_t cap, const Key& key, const Value& value) {
        size_t index = hash(key) % cap;
        while (target[index].occupied) {
            if (target[index].key == key) {
                target[index].value = value;
                return;
            }
            index = (index + 1) % cap;  // 线性探测
        }
        target[index].key = key;
        target[index].value = value;
        target[index].occupied = true;
    }

public:
    // 迭代器类
    class Iterator {
    private:
        Node* nodes;
        size_t capacity;
        size_t index;

        void advance() {
            while (index < capacity && !nodes[index].occupied) {
                ++index;
            }
        }

    public:
        Iterator(Node* n, size_t cap, size_t idx) : nodes(n), capacity(cap), index(idx) {
            advance();
        }

        struct Pair {
            Key key;
            Value& value;
            Pair(const Key& k, Value& v) : key(k), value(v) {}
        };

        Pair operator*() const {
            return Pair(nodes[index].key, nodes[index].value);
        }

        Iterator& operator++() {
            ++index;
            advance();
            return *this;
        }

        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const Iterator& other) const {
            return nodes == other.nodes && index == other.index;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    // std::map接口实现
    CustomMap(size_t initialCapacity = 16) : capacity(initialCapacity), size_(0) {
        nodes = new Node[capacity];
    }

    CustomMap(const CustomMap& other) : capacity(other.capacity), size_(other.size_) {
        nodes = new Node[capacity];
        for (size_t i = 0; i < capacity; ++i) {
            nodes[i] = other.nodes[i];
        }
    }

    ~CustomMap() {
        delete[] nodes;
    }

    CustomMap& operator=(const CustomMap& other) {
        if (this != &other) {
            delete[] nodes;
            capacity = other.capacity;
            size_ = other.size_;
            nodes = new Node[capacity];
            for (size_t i = 0; i < capacity; ++i) {
                nodes[i] = other.nodes[i];
            }
        }
        return *this;
    }

    void insert(const Key& key, const Value& value) {
        if (size_ >= capacity * 0.75) {
            resize(capacity * 2);
        }
        insertNode(nodes, capacity, key, value);
        size_++;
    }

    Value& operator[](const Key& key) {
        size_t index = findKey(key);
        if (!nodes[index].occupied) {
            insert(key, Value());
            size_++;
        }
        return nodes[index].value;
    }

    bool count(const Key& key) const {  // std::map::count
        return contains(key) ? 1 : 0;
    }

    bool contains(const Key& key) const {
        size_t index = findKey(key);
        return nodes[index].occupied && nodes[index].key == key;
    }

    void erase(const Key& key) {
        size_t index = findKey(key);
        if (nodes[index].occupied && nodes[index].key == key) {
            nodes[index].occupied = false;
            size_--;
        }
    }

    size_t size() const { return size_; }

    bool empty() const { return size_ == 0; }

    void clear() {
        for (size_t i = 0; i < capacity; ++i) {
            nodes[i].occupied = false;
        }
        size_ = 0;
    }

    Iterator begin() {
        return Iterator(nodes, capacity, 0);
    }

    Iterator end() {
        return Iterator(nodes, capacity, capacity);
    }

    Iterator find(const Key& key) {  // std::map::find
        size_t index = findKey(key);
        if (nodes[index].occupied && nodes[index].key == key) {
            return Iterator(nodes, capacity, index);
        }
        return end();
    }

private:
    size_t findKey(const Key& key) const {
        size_t index = hash(key) % capacity;
        while (nodes[index].occupied) {
            if (nodes[index].key == key) {
                return index;
            }
            index = (index + 1) % capacity;
        }
        return index;
    }

    // 自定义哈希函数，避免依赖<functional>
    size_t hash(const Key& key) const {
        return static_cast<size_t>(key);  // 假设Key是int，可根据需要调整
    }
};

// // 测试代码
// int TestMap() {
//     CustomMap<int, float> map;
//     map.insert(1, 1.0f);
//     map.insert(2, 2.0f);
//     map[3] = 3.0f;

//     printf("Map size: %zu\n", map.size());
//     printf("Contains 2: %d\n", map.contains(2));
//     printf("Count 2: %zu\n", map.count(2));

//     for (auto it = map.begin(); it != map.end(); ++it) {
//         auto pair = *it;
//         printf("Key: %d, Value: %.1f\n", pair.key, pair.value);
//     }

//     auto it = map.find(2);
//     if (it != map.end()) {
//         auto pair = *it;
//         printf("Found key 2 with value: %.1f\n", pair.value);
//     }

//     map.erase(2);
//     printf("After erase, size: %zu\n", map.size());

//     return 0;
// }
}