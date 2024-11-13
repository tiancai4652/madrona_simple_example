#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>

#include "2_types_mflow.hpp"
#include "init.hpp"
#include <assert.h>

namespace madsimple {

using madrona::Entity;
using madrona::CountT;




template <typename T1, typename T2>
class Map
{
private:
    // Pair<T1, T2> Entry = new Pair(T1 key, T2 value)
    class Entry{
        public:
        T1 key;
        T2 value;
        Entry() : key(T1()), value(T2()) {} // 使用T1和T2的默认构造函数
        Entry(const T1 &key, const T2 &value) : key(key), value(value) {}
    };
    Entry entries[MAP_SIZE];
    int length;

public:
    //key和value的自定义类都需要重载具有构造函数
    Map() : length(0)
    {
        for (int i = 0; i < MAP_SIZE; ++i)
        {
            entries[i] = Entry(); // 初始化每个Entry
        }
    }
    
    //需要在key的自定义类中重载比较符号==
    T2 &operator[](const T1 &key)
    {
        return get(key);
    }

    T2 &at(const T1 &key){
        return get(key);
    }

    Entry* find(const T1& key) {
        for (int i = 0; i < length; i++) {
            if (entries[i].key == key) {
                return &entries[i];  // 返回指向找到的键值对的指针
            }
        }
        return endEntry();  // 如果没有找到，返回 end()
    }

    Entry* endEntry() {
        return &entries[length];  // 返回数组末尾的指针，表示没有找到
    }

    void clear() {
        for (int i = 0; i < length; i++) {
            entries[i] = Entry();
        }
    }

    //先查找再索引：assert(map.find(key)); map[key]
    T2 &get(const T1 &key)
    {
        for (int i = 0; i < length; i++)
        {
            if (entries[i].key == key)
            {
                return entries[i].value;
            }
        }
        // 如果key不存在，添加新的条目
        assert(length < MAP_SIZE - 1);
        entries[length] = Entry(key, T2());
        return entries[length++].value;
    }

    void set(const T1 &key, const T2 &value)
    {
        for (int i = 0; i < length; i++)
        {
            if (entries[i].key == key)
            {
                entries[i].value = value;
                return;
            }
        }
        // 如果key不存在，添加新的条目
        assert(length < MAP_SIZE - 1);
        entries[length] = Entry(key, value);
        length++;
    }

    int size() const
    {
        return length;
    }

    bool remove(const T1& key) {
        for (int i = 0; i < length; i++) {
            if (entries[i].key == key) {
                // 找到了键，现在移除它
                // 通过将后面的元素向前移动来覆盖这个元素
                for (int j = i; j < length - 1; j++) {
                    entries[j] = entries[j + 1];
                }
                // 减少长度
                length--;
                // 由于我们的数组现在比实际长度小了一个元素，
                // 我们可以选择清理最后一个元素，但这不是必须的，
                // 因为它会在下一次添加或者到达那个位置时被覆盖。
                // 但为了保持数据的整洁，我们可以选择重置最后一个元素。
                entries[length] = Entry(); // 可选
                return true; // 表示删除成功
            }
        }
        return false; // 如果没有找到键，返回false
    }


    // 迭代器类，支持遍历键值对
    class Iterator {
        private:
        Entry* current;
        public:
        // 构造函数，初始化迭代器
        Iterator(Entry* start) : current(start) {}

        // 前缀++，移动到下一个元素
        Iterator& operator++() {
            ++current;
            return *this;
        }

        // 解引用，返回当前键值对
        Entry& operator*() const {
            return *current;
        }

        // 比较操作符，用于结束判断
        bool operator!=(const Iterator& other) const {
            return current != other.current;
        }

    };
    Iterator begin() {
        return Iterator(entries);
    }
    Iterator end() {
        return Iterator(entries + length);
    }

};


class Engine;

struct Sim : public madrona::WorldBase {
    struct Config {
        uint32_t maxEpisodeLength;
        bool enableViewer;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphBuilder &builder,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;
    const GridState *grid;
    uint32_t maxEpisodeLength;


    Entity inPorts[(K_ARY*K_ARY*5/4)*K_ARY]; 
    Entity ePorts[(K_ARY*K_ARY*5/4)*K_ARY];
    uint32_t numInPort;
    uint32_t numEPort;

    Entity _switches[K_ARY*K_ARY*5/4];
    uint32_t numSwitch;

    // Entity _snd_flows[2*K_ARY*K_ARY*K_ARY/4];
    // Entity _recv_flows[2*K_ARY*K_ARY*K_ARY/4];
    // uint32_t num_snd_flow;
    // uint32_t num_recv_flow;   

    // indexed by the flow_id
    // each npu has a map of snd_flows and a map of rcv_flows
    Map<uint16_t, Entity> snd_flows[K_ARY*K_ARY*K_ARY/4]; 
    Map<uint16_t, Entity> recv_flows[K_ARY*K_ARY*K_ARY/4];

    uint32_t flow_count[K_ARY*K_ARY*K_ARY/4]; // for flow_id

    Entity _nics[K_ARY*K_ARY*K_ARY/4];
    uint32_t num_nic;

    Entity _npus[K_ARY*K_ARY*K_ARY/4];
    uint32_t num_npu;  

};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}