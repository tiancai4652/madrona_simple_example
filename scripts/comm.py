import mmap
import posix_ipc
import struct




class MadronaMsg:
    # def __init__(self, data):
    #     self.type, self.event_id, self.time, self.src, self.dst, self.size = struct.unpack('6i', data)
    def __init__(self, type, event_id, time, src, dst, size):
        self.type = type
        self.event_id = event_id
        self.time = time
        self.src = src
        self.dst = dst
        self.size = size
    
    def __repr__(self):
        return f"MadronaMsg(type={self.type}, event_id={self.event_id}, time={self.time}, src={self.src}, dst={self.dst}, size={self.size})"
    def pack(self):
        # 根据属性顺序和数据类型，这里假设所有属性都是整数
        return struct.pack('6i', self.type, self.event_id, self.time, self.src, self.dst, self.size)



def _create_and_write_shared_memory(messages,map_file):
    num_messages = len(messages)
    header_size = struct.calcsize('i')  # 消息数量头部大小
    message_size = struct.calcsize('6i')  # 每个 MadronaMsg 的大小
    total_size = header_size + num_messages * message_size  # 总共需要的内存大小

    # # 创建或打开共享内存
    # memory_name = "myshm"
    # memory = posix_ipc.SharedMemory(memory_name, posix_ipc.O_CREX, size=total_size)

    # # 映射共享内存
    # map_file = mmap.mmap(memory.fd, total_size, mmap.MAP_SHARED, mmap.PROT_WRITE)

    # 写入消息数量到头部
    map_file.seek(0)
    map_file.write(struct.pack('i', num_messages))

    # 写入每个 MadronaMsg
    for msg in messages:
        map_file.write(msg.pack())

    # 清理
    # map_file.close()
    # posix_ipc.unlink_shared_memory(memory_name)



def comm():
    numMessages=10
    memory = posix_ipc.SharedMemory("myshm")
    size=24*numMessages+4
    map_file = mmap.mmap(memory.fd, size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
    # map_file = mmap.mmap(memory.fd, mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
    semaphore_a = posix_ipc.Semaphore("semA")
    semaphore_b = posix_ipc.Semaphore("semB")

    # 等待 C++ 程序的数据
    semaphore_a.acquire()
    numMessages, = struct.unpack('i', map_file.read(4))  # 读取消息数量
    messages = []
    for _ in range(numMessages):
        data = map_file.read(24)  # 每个 MadronaMsg 是 24 字节
        # message = MadronaMsg(data)
        unpacked_data = struct.unpack('6i', data)
        message = MadronaMsg(*unpacked_data)  # 使用解包调用构造函数
        messages.append(message)
    print(f"Python: Received from C++: {messages}")

    # 处理数据并写回
    messages = [
    MadronaMsg(1, 100, 5000, 10, 20, 30),
    MadronaMsg(2, 101, 6000, 11, 21, 31),
    MadronaMsg(3, 102, 7000, 12, 22, 32)
]
    _create_and_write_shared_memory(messages,map_file)
    # new_value = value + 100
    # map_file.seek(0)
    # map_file.write(struct.pack('i', new_value))
    print(f"Python: Data processed and written back: {messages}")

    # 通知 C++ 程序处理完成
    semaphore_b.release()

    # 清理资源
    map_file.close()
    memory.close_fd()
    semaphore_a.close()
    semaphore_b.close()
    
comm()