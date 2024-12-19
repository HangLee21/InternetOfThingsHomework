import numpy as np
import pygame
from scipy.io.wavfile import write
import sounddevice as sd

# 调制参数
sample_rate = 44100  # 采样率
bit_duration = 0.2  # 每个比特的持续时间（增加比特持续时间以减少ISI）
freq_0 = 1000  # 频率0对应1000 Hz
freq_1 = 3000  # 频率1对应3000 Hz，增加频率间隔以降低干扰
max_payload_length = 192  # 最大负载长度（比特数）

# 升余弦滤波器参数
alpha = 0.35  # 滤波器的滚降因子
span = 6  # 滤波器的长度，表示滤波器的脉冲持续时间


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
def create_packet(data, packet_rank, total_packets):
    preamble = '11111111'  # 前导码

    rank = format(packet_rank, '08b')  # 包头：包序号信息
    total_packet_length = format(total_packets, '08b')  # 包头：总包数（8 bit）

    # 对数据部分进行汉明编码
    encoded_data = hamming_encode(data)
    payload_length = format(len(encoded_data), '08b')  # 包头：数据长度（以比特为单位）
    # 组成数据包
    packet = preamble + payload_length + rank + total_packet_length + encoded_data

    print(packet)
    return packet


# 划分文本为多个数据包
def split_into_packets(binary_data):
    packets = []
    packet_rank = 1
    total_packets = (len(binary_data) + max_payload_length - 1) // max_payload_length  # 总包数
    # 每个数据包最大负载长度为96 bits
    for i in range(0, len(binary_data), max_payload_length):
        data_chunk = binary_data[i:i + max_payload_length]
        packet = create_packet(data_chunk, packet_rank, total_packets)
        packets.append(packet)
        packet_rank += 1
    return packets


# 升余弦滤波器（Raised Cosine Filter）
def raised_cosine_filter(beta, span, sps):
    t = np.linspace(-span / 2, span / 2, span * sps)
    h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
    h /= np.sum(h)
    return h


# 应用升余弦滤波
def apply_pulse_shaping(signal, sps):
    filter_taps = raised_cosine_filter(alpha, span, sps)
    shaped_signal = np.convolve(signal, filter_taps, mode='same')
    return shaped_signal


# FSK调制
def modulate_fsk(binary_data):
    sps = int(sample_rate * bit_duration)  # 每个比特的采样点数
    signal = []

    for bit in binary_data:
        t = np.linspace(0, bit_duration, sps, endpoint=False)
        if bit == '0':
            signal.append(np.sin(2 * np.pi * freq_0 * t))
        else:
            signal.append(np.sin(2 * np.pi * freq_1 * t))

    # 合并所有符号信号
    modulated_signal = np.concatenate(signal)

    # 应用脉冲成形（升余弦滤波）
    modulated_signal = apply_pulse_shaping(modulated_signal, sps)

    return modulated_signal


# 保存每个包的调制信号为不同的wav文件
def save_signal_to_wav(signal, packet_rank):
    filename = f'output/modulated_signal_packet_{packet_rank}.wav'
    write(filename, sample_rate, signal.astype(np.float32))


# 播放信号
def play_signal(signal):
    # 将信号的振幅缩放到[-1, 1]范围内，并转换为int16类型
    signal = np.array(signal)
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    try:
        sd.play(signal, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"音频播放出错: {e}")


# 主程序
def main():
    # 控制是否存储为 wav 文件或直接播放声音
    save_to_wav = False  # 设置为 True 来保存为 wav 文件，设置为 False 来播放声音

    text = input("请输入文本：")
    binary_data = text_to_binary(text)  # 将文本转为二进制
    packets = split_into_packets(binary_data)  # 将数据划分为多个包

    # 发送数据包
    total_packets = len(packets)  # 获取总包数
    for packet_rank, packet in enumerate(packets, 1):  # 枚举包的序号
        signal = modulate_fsk(packet)  # 调制为声波信号

        if save_to_wav:
            save_signal_to_wav(signal, packet_rank)  # 保存每个包的信号为不同的wav文件
        else:
            play_signal(signal)  # 播放声音

        # 判断是否为最后一个包，控制输出
        if packet_rank == total_packets:
            print("所有数据包已发送完毕。")


# 运行主程序
main()
