from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field
from parse_chakra import Node,Attribute,BoolList,Parse
from pathlib import Path

class NodeType(Enum):
    COMP_NODE = 1
    COMM_SEND_NODE = 2
    COMM_RECV_NODE=3
    COMM_COLL_NODE=4
    

class AttributeKey(Enum):
    comm_para = 1
    comm_size = 2
    comm_src = 3
    comm_dst = 4
    involved_dim=5

# 定义占位符
placeholder = 4294967295

def node_to_int_array(node: Node) -> List[int]:
    int_array = []

    # 将 name 转换为 20 个 int 类型
    name_ints = [ord(c) for c in node.name[:20]]  # 截取前20个字符的 ASCII 值
    name_ints += [placeholder] * (20 - len(name_ints))  # 不足 20 个字符则补 placeholder
    int_array.extend(name_ints)

    # 将 type 转换为 int 类型
    node_type = NodeType[node.type].value if node.type in NodeType.__members__ else placeholder
    int_array.append(node_type)

    # 将 id 转换为一个 int 类型
    node_id = int(node.id) if node.id is not None and node.id.isdigit() else placeholder
    int_array.append(node_id)

    # 确保 dataDeps 不是 None
    data_deps = node.dataDeps if node.dataDeps is not None else []
    # 将 data_deps 转换为 10 个 int 类型
    data_deps_ints = [int(dep) if dep.isdigit() else placeholder for dep in data_deps[:10]]
    data_deps_ints += [placeholder] * (10 - len(data_deps_ints))  # 不足 10 个则补 placeholder
    int_array.extend(data_deps_ints)

    # 将 attr 转换为 14 个 int 类型
    attr_ints = []
    for attr in node.attr:
        if attr.name not in AttributeKey.__members__:
            continue
        key_int = AttributeKey[attr.name].value
        value_int = placeholder
        if attr.boolVal is not None:
            value_int = int(attr.boolVal)
        elif attr.uint32Val is not None:
            value_int = attr.uint32Val
        elif attr.int64Val is not None:
            value_int = attr.int64Val
        elif attr.uint64Val is not None:
            value_int = attr.uint64Val
        elif attr.boolList is not None:
            bool_list_vals = [int(val) for val in attr.boolList.values[:3]]  # 取前3个值
            bool_list_vals += [placeholder] * (3 - len(bool_list_vals))  # 不足3个则补 placeholder
            attr_ints.extend([key_int, placeholder] + bool_list_vals)  # placeholder表示占位符
            continue
        attr_ints.extend([key_int, value_int])
        if len(attr_ints) >= 14:
            break  # 只取前 7 个键值对

    attr_ints += [placeholder] * (14 - len(attr_ints))  # 不足 14 个则补 placeholder
    int_array.extend(attr_ints)

    # 将 durationMicros 添加到数组末尾
    int_array.append(node.durationMicros)

    return int_array

def int_array_to_node(int_array: List[int]) -> Node:
    # 将 int 数组转换回 Node 对象
    name_chars = [chr(i) for i in int_array[:20] if i != placeholder]
    name = ''.join(name_chars)

    node_type = NodeType(int_array[20]).name if int_array[20] in NodeType._value2member_map_ else 'UNKNOWN'
    node_id = str(int_array[21]) if int_array[21] != placeholder else None

    data_deps = [str(i) for i in int_array[22:32] if i != placeholder]

    attrs = []
    for i in range(32, 46, 4):
        if int_array[i] == placeholder and int_array[i+1] == placeholder:
            continue
        key_name = AttributeKey(int_array[i]).name if int_array[i] in AttributeKey._value2member_map_ else 'UNKNOWN'
        if int_array[i+1] == placeholder:
            bool_list_vals = int_array[i+2:i+5]
            attr = Attribute(name=key_name, boolList=BoolList(values=[bool(val) for val in bool_list_vals if val != placeholder]))
        else:
            attr = Attribute(name=key_name, uint32Val=int_array[i+1])
        attrs.append(attr)

    duration_micros = int_array[46]

    return Node(
        name=name,
        type=node_type,
        id=node_id,
        dataDeps=data_deps,
        attr=attrs,
        durationMicros=duration_micros
    )
 
# 测试示例
def test_conversion():
    json_file_path = '/home/zhangran/madrona2/2/madrona_simple_example/scripts/input/npu.0.json'
    nodes = Parse(json_file_path)
    
    if nodes:
        first_node = nodes[1]
        print(f"转换前的 Node 对象: {first_node}")
        int_array = node_to_int_array(first_node)
        print(f"转换后的整数数组: {int_array}")

        node_back = int_array_to_node(int_array)
        print(f"转换回的 Node 对象: {node_back}")

# 执行测试
# test_conversion()

def nodes_to_int_array(json_file_path):
    # json_file_path = '/home/zhangran/madrona2/2/madrona_simple_example/scripts/input/npu.0.json'
    nodes = Parse(json_file_path)
    result=[]
    if nodes:
        for node in nodes:
            result.extend(node_to_int_array(node))
    return result
 
 
def folder_to_int_array(folder_path):
    # folder_path = '/home/zhangran/madrona2/2/madrona_simple_example/scripts/input'
    result=[]
    folder = Path(folder_path)
    file_paths = [str(file) for file in folder.rglob('*') if file.is_file()]
    for file in file_paths:
        result.append(nodes_to_int_array(file))
    return result
 