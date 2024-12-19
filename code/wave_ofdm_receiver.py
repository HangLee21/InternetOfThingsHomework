import numpy as np
import pyaudio
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox
import struct
import time


# QAM符号转二进制比特串
def qam_to_binary(qam_symbols, M):
    binary_data = ''
    bits_per_symbol = int(np.log2(M))  # 每个符号的比特数
    for symbol in qam_symbols:
        binary_value = format(symbol, f'0{bits_per_symbol}b')
        binary_data += binary_value
    return binary_data


# 二进制数据转文本
def binary_to_text(binary_data):
    text = ''
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i + 8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text


# 前导码检测
def detect_preamble(signal, fs, preamble_bits, bit_duration, threshold=0.5):
    """使用前导码进行信号检测"""
    num_samples_per_bit = int(fs * bit_duration)  # 每个比特的持续时间（以样本数表示）
    preamble_samples = len(preamble_bits) * num_samples_per_bit  # 前导码长度（样本数）

    # 创建前导码（'11111111'）
    preamble_signal = np.array([int(bit) for bit in preamble_bits])

    # 计算信号与前导码的相关性
    correlation = np.correlate(signal, preamble_signal, mode='valid')
    peak_idx = np.argmax(np.abs(correlation))  # 找到相关性峰值的位置

    # 如果相关性超过阈值，则认为找到了前导码
    if np.max(np.abs(correlation)) > threshold:
        return peak_idx  # 返回检测到的前导码位置
    else:
        return -1  # 没有检测到前导码


# 解调信号（QAM）
def demodulate_signal(signal, fs, M, symbol_duration):
    """解调QAM调制的信号"""
    symbols = []
    num_samples_per_symbol = int(fs * symbol_duration)
    num_symbols = len(signal) // num_samples_per_symbol

    for i in range(num_symbols):
        start_idx = i * num_samples_per_symbol
        end_idx = (i + 1) * num_samples_per_symbol
        symbol_signal = signal[start_idx:end_idx]

        # 简单的频率估计方法
        freqs = np.fft.fftfreq(len(symbol_signal), 1 / fs)
        spectrum = np.abs(np.fft.fft(symbol_signal))
        peak_freq_idx = np.argmax(spectrum)
        symbol_freq = freqs[peak_freq_idx]

        # 将频率映射到QAM符号
        symbol = int(np.round(symbol_freq % M))
        symbols.append(symbol)

    binary_data = qam_to_binary(symbols, M)
    return binary_data

def parse_header(header_bits):
    """解析包头，返回payload_length, packet_rank, total_packet_length"""
    payload_length = int(header_bits[:8], 2)
    packet_rank = int(header_bits[8:16], 2)
    total_packet_length = int(header_bits[16:], 2)
    return payload_length, packet_rank, total_packet_length


def decode_data_packet(payload_bits):
    """解码数据包的内容（payload），返回解码的文本"""
    return ''.join(chr(int(payload_bits[i:i+8], 2)) for i in range(0, len(payload_bits), 8))


def receive_signal_from_microphone(fs, preamble_bits, bit_duration, symbol_duration):
    buffer = []
    sample_rate = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,  # 采样频率（Hz）
                    input=True,
                    frames_per_buffer=1024)
    print("开始接收音频信号...")

    # 用于跟踪接收的包序号
    total_packets_received = 0
    total_packets = 0  # 先设置为 0，稍后通过解析 header 获取总包数

    try:
        while True:
            data = stream.read(1024)
            buffer.extend(np.frombuffer(data, dtype=np.int16))  # 将新数据添加到缓冲区

            # 检查是否有足够的数据来解码
            if len(buffer) > sample_rate * bit_duration:
                signal_bits = demodulate_signal(buffer, M=4, symbol_duration=symbol_duration, fs=fs)  # 解调OFDM信号
                preamble = '11111111'  # 前导码
                preamble_idx = signal_bits.find(preamble)

                if preamble_idx != -1:
                    print("检测到前导码，开始解码数据包...")
                    print(signal_bits)

                    # 在前导码下画波浪线
                    line_length = len(signal_bits)
                    wave_line = " " * preamble_idx + "~" * len(preamble) + " " * (
                                line_length - (preamble_idx + len(preamble)))
                    print(wave_line)

                    # 等待足够的header数据
                    header_end_idx = preamble_idx + len(preamble) + 24  # header为24位
                    if len(signal_bits) >= header_end_idx:
                        header_bits = signal_bits[preamble_idx + len(preamble):header_end_idx]
                        # 解析header
                        payload_length, packet_rank, total_packet_length = parse_header(header_bits)
                        total_packets = total_packet_length  # 获取总包数
                        print(f"包头解析：payload_length={payload_length}, packet_rank={packet_rank}, total_packet_length={total_packets}")

                        # 等待足够的payload数据
                        payload_end_idx = header_end_idx + payload_length
                        if len(signal_bits) >= payload_end_idx:
                            payload_bits = signal_bits[header_end_idx: payload_end_idx]

                            # 解码数据包
                            text = decode_data_packet(payload_bits)
                            print(f"包{packet_rank}内容：{text}")

                            # 将解码内容缓存（或者直接处理）
                            buffer = buffer[preamble_idx + len(preamble) + 24 + payload_length:]

                            # 更新接收的包的序号
                            total_packets_received += 1
                            # 如果接收到所有包，停止接收
                            if total_packets_received == total_packets:
                                print("所有包已接收完毕，停止接收数据。")
                                break

                # 如果缓冲区没有足够的数据，则继续接收
                else:
                    continue
    except KeyboardInterrupt:
        print("停止接收音频信号")
        stream.stop_stream()
        stream.close()
        p.terminate()


# 创建可视化界面
def create_gui(decoded_text):
    """创建GUI展示解码结果"""
    window = tk.Tk()
    window.title("音频信号解码")

    label = tk.Label(window, text="解码结果:", font=('Arial', 14))
    label.pack(pady=10)

    text_box = tk.Text(window, height=10, width=50, wrap=tk.WORD)
    text_box.pack(pady=10)
    text_box.insert(tk.END, decoded_text)

    button = tk.Button(window, text="关闭", command=window.quit, font=('Arial', 12))
    button.pack(pady=10)

    window.mainloop()


# 主函数
def main():
    fs = 44100  # 采样频率
    preamble_bits = '11111111'  # 前导码
    bit_duration = 0.1  # 每个比特的持续时间（秒）
    symbol_duration = 0.1  # 每个符号的持续时间（秒）
    M = 4  # 4-QAM（QPSK）

    print("等待接收信号...")
    receive_signal_from_microphone(fs, preamble_bits, bit_duration, symbol_duration)

    # if decoded_text:
    #     print("解码成功，结果为：", decoded_text)
    #     create_gui(decoded_text)  # 显示解码结果


if __name__ == "__main__":
    main()
