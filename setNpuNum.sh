#!/bin/bash

# 启用调试模式
set -x

# 提示用户输入 npu_num
read -p "请输入 npu_num 的值: " npu_num

# 检查用户是否输入了 npu_num
if [[ -z "$npu_num" ]]; then
  echo "未输入 npu_num，程序退出。"
  exit 1
fi

# 定义文件路径和替换规则
declare -A replacements
replacements["scripts/run.py"]="s/chakra_nodes_num\s*=\s*1/chakra_nodes_num=${npu_num}/g"
replacements["src/mgr.cpp"]="s/\\{impl_->cfg.numWorlds,\\s*1,\\s*10000000\\}/\\{impl_->cfg.numWorlds, ${npu_num}, 10000000\\}/g"
replacements["src/types.hpp"]="s/uint32_t[[:space:]]\\+data\\[1\\]\\[10000000\\]/uint32_t data[${npu_num}][10000000]/g"
# 遍历文件并执行替换
for file in "${!replacements[@]}"; do
  if [[ -f "$file" ]]; then
    echo "处理文件: $file"
    
    # # 打印替换前的内容
    # echo "替换前的内容:"
    # cat "$file"

    # 测试正则表达式匹配
    echo "测试正则表达式匹配:"
    grep -E "${replacements[$file]}" "$file"

    # 替换内容
    sed -i -E "${replacements[$file]}" "$file"

    # # 打印替换后的内容
    # echo "替换后的内容:"
    # cat "$file"
  else
    echo "文件未找到: $file"
  fi
done

echo "替换完成！"