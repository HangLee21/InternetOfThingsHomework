import numpy as np
import pygame
from scipy.io.wavfile import write

# 调制参数
sample_rate = 44100  # 采样率
bit_duration = 0.1   # 每个比特的持续时间
freq_0 = 1000        # 频率0对应1000 Hz
freq_1 = 2000        # 频率1对应2000 Hz
max_payload_length = 96  # 最大负载长度（比特数）

"""
struct
preamble: 11111111
header: payload_length(8 bit) payload_rank(8 bit)
payload: max_length = 96 bit
"""

# 转换文本为二进制比特串
def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

# 创建数据包
def create_packet(data, packet_rank):
    preamble = '11111111'  # 前导码
    length = format(len(data), '08b')  # 包头：长度信息
    rank = format(packet_rank, '08b')  # 包头：包序号信息
    packet = preamble + length + rank + data  # 数据包结构
    return packet

# 划分文本为多个数据包
def split_into_packets(binary_data):
    packets = []
    packet_rank = 1
    # 每个数据包最大负载长度为96 bits
    for i in range(0, len(binary_data), max_payload_length):
        data_chunk = binary_data[i:i+max_payload_length]
        packet = create_packet(data_chunk, packet_rank)
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

# 模拟发送信号，通过Pygame播放音频
def play_signal(signal):
    # 将信号转换为合适的音频格式，范围从 -1 到 1
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    # 使用pygame播放信号
    pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
    pygame.mixer.Sound(signal.tobytes()).play()
    # 播放直到完成
    pygame.time.wait(int(len(signal) / sample_rate * 1000))

# 主程序
def main():
    text = input("请输入文本：")
    binary_data = text_to_binary(text)  # 将文本转为二进制
    packets = split_into_packets(binary_data)  # 将数据划分为多个包
    for packet_rank, packet in enumerate(packets, 1):  # 枚举包的序号
        signal = modulate_fsk(packet)  # 调制为声波信号
        print(f"正在发送包 {packet_rank}...")
        play_signal(signal)  # 模拟发送信号

# 运行主程序
main()