#pragma once

#include <string>
#include <vector>

namespace AstraSim {

// 定义一个简单的数据结构用于模拟
struct SimulationData {
    float time;
    float value;
    int description[1024];
};

// SystemLayer 基类
class SystemLayer {
public:
    SystemLayer();
    virtual ~SystemLayer();

    // 基本方法
    virtual void initialize();
    virtual void update(float deltaTime);
    virtual void shutdown();
    
    // 模拟相关方法
    virtual void runSimulation(float duration);
    virtual SimulationData getSimulationResult() const;
    
    // 状态查询方法
    virtual bool isInitialized() const;
    virtual std::string getStatus() const;
    
    // 配置方法
    virtual void setParameter(const std::string& name, float value);
    virtual float getParameter(const std::string& name) const;

protected:
    bool initialized_;
    std::string status_;
    std::vector<std::pair<std::string, float>> parameters_;
    SimulationData currentResult_;
};

} // namespace AstraSim 