# 完整的解码流程
import os

import numpy as np
import tkinter as tk
from scipy.io.wavfile import read

# 解调参数
sample_rate = 44100  # 采样率
bit_duration = 0.1  # 每个比特的持续时间
freq_0 = 1000  # 频率0对应1000 Hz
freq_1 = 2000  # 频率1对应2000 Hz
max_payload_length = 96  # 最大负载长度（比特数）


# 读取声波信号并提取数据
def read_wave(file_path):
    rate, data = read(file_path)
    if len(data.shape) == 2:  # 如果是立体声，将其转换为单声道
        data = data.mean(axis=1)
    return rate, data


# 提取信号中的频率成分（FSK解调）
def demodulate_fsk(signal):
    signal = np.array(signal)
    # 计算每个比特的FFT
    signal_length = len(signal)
    bit_count = int(signal_length / (sample_rate * bit_duration))
    decoded_bits = []

    for i in range(bit_count):
        start_idx = int(i * sample_rate * bit_duration)
        end_idx = int((i + 1) * sample_rate * bit_duration)
        segment = signal[start_idx:end_idx]

        # 计算频率
        freqs = np.fft.fftfreq(len(segment), 1 / sample_rate)
        fft_vals = np.fft.fft(segment)
        peak_freq = freqs[np.argmax(np.abs(fft_vals))]

        # 判断频率属于0或1
        if peak_freq < (freq_0 + freq_1) / 2:
            decoded_bits.append('0')
        else:
            decoded_bits.append('1')

    # 获取解调后的比特串
    signal_bits = ''.join(decoded_bits)
    print(f"解调后的比特串: {signal_bits}")

    return signal_bits


# 解码二进制比特串为字符
def binary_to_text(binary_data):
    text = ''
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i + 8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text


# 拼接多个数据包的负载内容
def concatenate_payloads(payloads):
    return ''.join(payloads)


# GUI界面展示解码结果
def display_decoded_text(decoded_text):
    window = tk.Tk()
    window.title("解码结果")

    label = tk.Label(window, text="解码后的文本：", font=("Arial", 14))
    label.pack(pady=10)

    result_label = tk.Label(window, text=decoded_text, font=("Arial", 12))
    result_label.pack(pady=20)

    close_button = tk.Button(window, text="关闭", command=window.quit)
    close_button.pack(pady=10)

    window.mainloop()


# 完整的解码流程
def decode_signal(file_paths):
    decoded_payloads = {}
    packet_rank = 1

    # 处理每个数据包文件
    for file_path in file_paths:
        # 读取音频文件
        rate, signal = read_wave(file_path)

        # 解调得到比特串
        signal_bits = demodulate_fsk(signal)

        # 假设前导码是'11111111'，并寻找前导码同步信号
        preamble = '11111111'  # 前导码
        preamble_idx = signal_bits.find(preamble)

        if preamble_idx == -1:
            print(f"未找到前导码，解码失败：{file_path}")
            continue

        # 解码数据包
        while preamble_idx != -1:
            # 获取包头的长度信息（8 bit）和包序号信息（8 bit）
            payload_length = int(signal_bits[preamble_idx + 8:preamble_idx + 16], 2)
            packet_rank = int(signal_bits[preamble_idx + 16:preamble_idx + 24], 2)  # 解析包序号
            print(f"包{packet_rank} 数据长度: {payload_length} bits")

            # 获取数据段（Payload）并将其添加到payload字典
            payload = signal_bits[preamble_idx + 24:preamble_idx + 24 + payload_length]
            decoded_payloads[packet_rank] = payload  # 将payload按照packet_rank存入字典

            # 查找下一个数据包的前导码
            preamble_idx = signal_bits.find(preamble, preamble_idx + 24 + payload_length)

    # 根据包序号排序所有数据包
    sorted_payloads = [decoded_payloads[rank] for rank in sorted(decoded_payloads.keys())]

    # 拼接所有数据包的内容
    full_data = concatenate_payloads(sorted_payloads)
    decoded_text = binary_to_text(full_data)

    print(decoded_text)
    # 展示解码后的文本
    display_decoded_text(decoded_text)


# 示例：执行解码流程
# 假设文件名为 modulated_signal_packet_1.wav, modulated_signal_packet_2.wav 等
# 获取指定目录下的所有.wav文件
directory = 'output'
file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
decode_signal(file_paths)
