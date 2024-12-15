import numpy as np
import pygame
from scipy.io.wavfile import write

# 调制参数
sample_rate = 44100  # 采样率
bit_duration = 0.1  # 每个比特的持续时间
freq_0 = 1000  # 频率0对应1000 Hz
freq_1 = 2000  # 频率1对应2000 Hz
max_payload_length = 96  # 最大负载长度（比特数）

"""
数据包结构:
preamble: 11111111
header: payload_length(8 bit) payload_rank(8 bit) end_flag(8 bit)
payload: 最大长度 = 96 bit
"""

# 汉明码编码器（7,4编码）
def hamming_encode(data):
    """采用汉明(7,4)编码，将每4位数据编码成7位（3个冗余位）"""
    encoded = []
    for i in range(0, len(data), 4):
        block = data[i:i + 4]
        if len(block) < 4:
            block = block + '0' * (4 - len(block))  # 补充不足的位

        # 分割成数据位和冗余位
        d1, d2, d3, d4 = block

        # 计算校验位（冗余位）
        p1 = int(d1) ^ int(d2) ^ int(d4)  # 校验位p1
        p2 = int(d1) ^ int(d3) ^ int(d4)  # 校验位p2
        p3 = int(d2) ^ int(d3) ^ int(d4)  # 校验位p3

        # 构建7位的编码块
        encoded.append(f'{p1}{p2}{d1}{p3}{d2}{d3}{d4}')

    return ''.join(encoded)


# 转换文本为二进制比特串
def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)


# 创建数据包
def create_packet(data, packet_rank, is_end=False):
    preamble = '11111111'  # 前导码
    rank = format(packet_rank, '08b')  # 包头：包序号信息
    end_flag = '1' if is_end else '0'  # 是否结束的标识符（8 bit）

    # 对数据部分进行汉明编码
    encoded_data = hamming_encode(data)
    length = format(len(encoded_data), '08b')  # 包头：长度信息
    # 组成数据包
    packet = preamble + length + rank + end_flag + encoded_data
    return packet


# 划分文本为多个数据包
def split_into_packets(binary_data):
    packets = []
    packet_rank = 1
    total_packets = (len(binary_data) + max_payload_length - 1) // max_payload_length  # 总包数
    # 每个数据包最大负载长度为96 bits
    for i in range(0, len(binary_data), max_payload_length):
        data_chunk = binary_data[i:i + max_payload_length]
        is_end = (packet_rank == total_packets)  # 判断是否是最后一个包
        packet = create_packet(data_chunk, packet_rank, is_end)
        packets.append(packet)
        packet_rank += 1
    return packets


# FSK调制
def modulate_fsk(binary_data):
    signal = []
    for bit in binary_data:
        t = np.linspace(0, bit_duration, int(sample_rate * bit_duration), endpoint=False)
        if bit == '0':
            signal.append(np.sin(2 * np.pi * freq_0 * t))
        else:
            signal.append(np.sin(2 * np.pi * freq_1 * t))
    return np.concatenate(signal)


# 保存每个包的调制信号为不同的wav文件
def save_signal_to_wav(signal, packet_rank):
    filename = f'output/modulated_signal_packet_{packet_rank}.wav'
    write(filename, sample_rate, signal.astype(np.float32))


# 播放信号
def play_signal(signal):
    pygame.mixer.init(frequency=sample_rate)
    pygame.mixer.Sound(np.array(signal * 32767, dtype=np.int16)).play()


# 主程序
def main():
    # 控制是否存储为 wav 文件或直接播放声音
    save_to_wav = False  # 设置为 True 来保存为 wav 文件，设置为 False 来播放声音

    text = input("请输入文本：")
    binary_data = text_to_binary(text)  # 将文本转为二进制
    packets = split_into_packets(binary_data)  # 将数据划分为多个包

    # 发送数据包
    for packet_rank, packet in enumerate(packets, 1):  # 枚举包的序号
        signal = modulate_fsk(packet)  # 调制为声波信号

        if save_to_wav:
            save_signal_to_wav(signal, packet_rank)  # 保存每个包的信号为不同的wav文件
        else:
            play_signal(signal)  # 播放声音


# 运行主程序
main()
