import mmap
import posix_ipc
import struct
from input_helper import MadronaEvent as MadronaMsg


# init
numMessages=1
memory = posix_ipc.SharedMemory("myshm")
size=28*numMessages+4
map_file = mmap.mmap(memory.fd, size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
# map_file = mmap.mmap(memory.fd, mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
semaphore_a = posix_ipc.Semaphore("semA")
semaphore_b = posix_ipc.Semaphore("semB")


def _create_and_write_shared_memory(messages,map_file):
    num_messages = len(messages)
    header_size = struct.calcsize('i')  # 消息数量头部大小
    message_size = struct.calcsize('7i')  # 每个 MadronaMsg 的大小

    # 写入消息数量到头部
    map_file.seek(0)
    map_file.write(struct.pack('i', num_messages))

    # 写入每个 MadronaMsg
    for msg in messages:
        map_file.write(msg.pack())

def receive_data():
    # wait msg
    semaphore_a.acquire()
    map_file.seek(0)
    # read msg num
    numMessages, = struct.unpack('i', map_file.read(4))  
    messages = []
    for _ in range(numMessages):
        data = map_file.read(24)  #  MadronaMsg = 24byte
        # message = MadronaMsg(data)
        unpacked_data = struct.unpack('6i', data)
        message = MadronaMsg(*unpacked_data)  
        messages.append(message)
    # print(f"Madrona: Received from Astrasim: {messages}")
    return messages

def send_data(messages):
    _create_and_write_shared_memory(messages,map_file)
    # print(f"Madrona: Data written to Astrasim: {messages}")
    # notifies astrasim
    semaphore_b.release()
    # semaphore_a.acquire()

# def comm():
#     data= receive_data()
#     # 处理数据并写回
#     messages = [MadronaMsg(1, 100, 5000, 10, 20, 30)]
#     _create_and_write_shared_memory(messages,map_file)
#     print(f"Madrona: Data written to Astrasim: {messages}")
#     # 通知 C++ 程序处理完成
#     semaphore_b.release()

    

def comm_dispose():
    map_file.close()
    memory.close_fd()
    semaphore_a.close()
    semaphore_b.close()

# comm()