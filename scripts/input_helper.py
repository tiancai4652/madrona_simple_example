import numpy as np
import torch


class MadronaEvent:
    def __init__(self, type, eventId, time, src, dst, size):
        self.type = type
        self.eventId = eventId
        self.time = time
        self.src = src
        self.dst = dst
        self.size = size
class MadronaEvents:
    def __init__(self, events):
        self.events = events

def madrona_events_to_int_array(madrona_events):
    int_array = []
    for event in madrona_events.events:
        int_array.extend([
            event.type,
            event.eventId,
            event.time,
            event.src,
            event.dst,
            event.size
        ])
    return int_array
def int_array_to_madrona_events(int_array):
    int_array = int_array[0]  # 获取一维张量数组
    events = []
    for i in range(0, len(int_array), 6):
        if i + 5 >= len(int_array):  # 如果不足6个元素则停止
            break
        if int_array[i] == 0 and int_array[i+1] == 0 and int_array[i+2] == 0 and \
           int_array[i+3] == 0 and int_array[i+4] == 0 and int_array[i+5] == 0:
            continue  # 忽略所有字段都为0的事件
        event = MadronaEvent(
            int_array[i],
            int_array[i+1],
            int_array[i+2],
            int_array[i+3],
            int_array[i+4],
            int_array[i+5]
        )
        events.append(event)
    return MadronaEvents(events)


def string_to_tensor(s, max_len=1000):
    """
    将字符串编码为数值，并存储在张量中
    :param s: 要编码的字符串
    :param max_len: 张量的最大长度
    :return: 包含字符串编码的张量
    """
    # 将字符串转换为ASCII编码
    encoded = [ord(c) for c in s]
    # 创建张量并填充编码值
    tensor = torch.zeros(max_len, dtype=torch.int32)
    tensor[:len(encoded)] = torch.tensor(encoded, dtype=torch.int32)
    return tensor
def tensor_to_string(tensor):
    """
    将编码张量转换回字符串
    :param tensor: 包含字符串编码的张量
    :return: 解码后的字符串
    """
    encoded = tensor.cpu().numpy().tolist()  # 先将张量复制到 CPU
    s = ''.join(chr(int(i)) for i in encoded if int(i) != 0)  # 确保元素为整数
    return s

def events_to_tensor(int_array, max_len=1000):
    # 将字符串转换为ASCII编码
    encoded = int_array
    # 创建张量并填充编码值
    tensor = torch.zeros(max_len, dtype=torch.int32)
    tensor[:len(encoded)] = torch.tensor(encoded, dtype=torch.int32)
    return tensor
def tensor_to_events(tensor):
    encoded = tensor.cpu().numpy().tolist() 
    return encoded