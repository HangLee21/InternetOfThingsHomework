import numpy as np
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
def detect_preamble(signal, fs, preamble_freq, preamble_length, threshold=0.5):
    """使用前导码进行信号检测"""
    num_samples_per_symbol = int(fs * 0.1)  # 每个符号的持续时间为0.1秒
    preamble_samples = int(preamble_length * fs)  # 前导码长度的样本数
    preamble_signal = np.zeros(preamble_samples)

    # 创建前导码（假设是某个频率的纯正弦波）
    t = np.arange(preamble_samples) / fs
    preamble_signal = np.cos(2 * np.pi * preamble_freq * t)  # 假设前导码为正弦波

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


# 从麦克风接收并解调音频信号
def receive_signal_from_microphone(fs, duration, preamble_freq, preamble_length, M, symbol_duration):
    """从麦克风接收信号，进行解调并展示解码结果"""
    # 录制音频
    print("正在录制音频信号...")
    recorded_signal = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # 等待录制完成

    # 进行前导码检测
    preamble_idx = detect_preamble(recorded_signal.flatten(), fs, preamble_freq, preamble_length)
    if preamble_idx == -1:
        print("未检测到前导码，信号同步失败！")
        return None

    print(f"前导码检测到，开始解调，前导码位置：{preamble_idx}")

    # 截取从前导码之后的信号
    signal_after_preamble = recorded_signal.flatten()[preamble_idx:]

    # 解调信号
    binary_data = demodulate_signal(signal_after_preamble, fs, M, symbol_duration)

    # 转换为文本
    decoded_text = binary_to_text(binary_data)
    return decoded_text


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
    preamble_freq = 1000  # 前导码频率（Hz）
    preamble_length = 1  # 前导码长度（秒）
    symbol_duration = 0.1  # 每个符号的持续时间（秒）
    M = 4  # 4-QAM（QPSK）

    print("等待接收信号...")
    decoded_text = receive_signal_from_microphone(fs, 5.0, preamble_freq, preamble_length, M, symbol_duration)

    if decoded_text:
        print("解码成功，结果为：", decoded_text)
        create_gui(decoded_text)  # 显示解码结果


if __name__ == "__main__":
    main()
