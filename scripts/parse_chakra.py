import json
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any

@dataclass
class BoolList:
    values: List[bool]

@dataclass
class Attribute:
    name: str
    boolVal: Optional[bool] = None
    uint32Val: Optional[int] = None
    boolList: Optional[BoolList] = None
    int64Val: Optional[int] = None
    uint64Val: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.int64Val, str):
            try:
                self.int64Val = int(self.int64Val)
            except (ValueError, TypeError):
                print(f"警告：无法将 int64Val '{self.int64Val}' 转换为整数。")
                self.int64Val = None
        if isinstance(self.uint64Val, str):
            try:
                self.uint64Val = int(self.uint64Val)
            except (ValueError, TypeError):
                print(f"警告：无法将 uint64Val '{self.uint64Val}' 转换为整数。")
                self.uint64Val = None

@dataclass
class Node:
    name: str
    type: str
    durationMicros: int
    attr: List[Attribute]
    id: Optional[str] = None
    dataDeps: Optional[List[str]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.durationMicros, str):
            try:
                self.durationMicros = int(self.durationMicros)
            except (ValueError, TypeError):
                print(f"警告：无法将 durationMicros '{self.durationMicros}' 转换为整数。")
                self.durationMicros = 0

# --- 解析 JSON 的辅助函数 ---
# (保持和之前一样的定义)

def parse_attribute(attr_data: dict) -> Attribute:
    """将字典解析为 Attribute 对象"""
    name = attr_data.get("name")
    bool_val = attr_data.get("boolVal")
    uint32_val = attr_data.get("uint32Val")
    int64_val = attr_data.get("int64Val")
    uint64_val = attr_data.get("uint64Val")
    bool_list_data = attr_data.get("boolList")

    bool_list_obj = None
    if bool_list_data and isinstance(bool_list_data, dict):
        values = bool_list_data.get("values")
        if isinstance(values, list):
            bool_list_obj = BoolList(values=values)

    return Attribute(
        name=name,
        boolVal=bool_val,
        uint32Val=uint32_val,
        boolList=bool_list_obj,
        int64Val=int64_val, # 传递给 __post_init__ 处理
        uint64Val=uint64_val # 传递给 __post_init__ 处理
    )

def parse_node(node_data: dict) -> Node:
    """将字典解析为 Node 对象"""
    attributes = [parse_attribute(attr) for attr in node_data.get("attr", [])]

    return Node(
        id=node_data.get("id"),
        name=node_data.get("name"),
        type=node_data.get("type"),
        dataDeps=node_data.get("dataDeps"),
        durationMicros=node_data.get("durationMicros"), # 传递给 __post_init__ 处理
        attr=attributes
    )

def parse_json_data(json_string: str) -> List[Node]:
    """解析 JSON 字符串并返回 Node 对象列表"""
    try:
        data = json.loads(json_string)
        if not isinstance(data, list):
            raise ValueError("JSON 顶层结构不是列表")
        return [parse_node(node_data) for node_data in data]
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return []
    except Exception as e:
        print(f"解析过程中发生错误: {e}")
        return []

def test():
    # 1. 定义 JSON 文件路径
    #    请将 'your_file.json' 替换为你的实际文件名和路径
    json_file_path = '/home/zhangran/madrona2/2/madrona_simple_example/scripts/input/npu.0.json'
    parsed_nodes = [] # 初始化为空列表

    # 2. 读取文件内容并解析
    try:
        # 使用 'with' 语句确保文件被正确关闭
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_string_from_file = f.read()
            # 调用之前的解析函数
            parsed_nodes = parse_json_data(json_string_from_file)

    except FileNotFoundError:
        print(f"错误：找不到文件 '{json_file_path}'")
    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")

    # 3. (如果解析成功) 打印解析结果进行验证
    if parsed_nodes:
        print(f"成功从 '{json_file_path}' 加载并解析了 {len(parsed_nodes)} 个节点。\n")
        for i, node in enumerate(parsed_nodes):
            print(f"--- Node {i} ---")
            print(f"ID: {node.id}")
            print(f"Name: {node.name}")
            print(f"Type: {node.type}")
            print(f"Duration (micros): {node.durationMicros} (Type: {type(node.durationMicros)})")
            print(f"Data Dependencies: {node.dataDeps}")
            print("Attributes:")
            for attr in node.attr:
                value = None
                value_type = None
                if attr.boolVal is not None:
                    value = attr.boolVal
                    value_type = type(value)
                elif attr.uint32Val is not None:
                    value = attr.uint32Val
                    value_type = type(value)
                elif attr.int64Val is not None:
                    value = attr.int64Val
                    value_type = type(value) # 验证是否为 int
                elif attr.uint64Val is not None:
                    value = attr.uint64Val
                    value_type = type(value) # 验证是否为 int
                elif attr.boolList is not None:
                    value = attr.boolList.values
                    value_type = f"BoolList({type(value)})"

                print(f"  - Name: {attr.name}, Value: {value} (Type: {value_type})")
            print("-" * 20)

        # 访问特定节点和属性的示例
        if len(parsed_nodes) > 0:
            first_node = parsed_nodes[0]
            print(f"\n第一个节点的名称: {first_node.name}")
            if len(first_node.attr) > 2 and first_node.attr[2].boolList:
                print(f"第一个节点第三个属性的 boolList 值: {first_node.attr[2].boolList.values}")
    else:
        print("未能成功解析节点数据。")
                
def Parse(json_file_path):
    try:
        # 使用 'with' 语句确保文件被正确关闭
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_string_from_file = f.read()
            # 调用之前的解析函数
            parsed_nodes = parse_json_data(json_string_from_file)
            return parsed_nodes
    except FileNotFoundError:
        print(f"错误：找不到文件 '{json_file_path}'")
    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")
        
# 2 int