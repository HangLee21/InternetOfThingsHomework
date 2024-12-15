import numpy as np
import pyaudio
import tkinter as tk
from scipy.fft import fft, fftfreq

# 解调参数
sample_rate = 44100  # 采样率
bit_duration = 0.1  # 每个比特的持续时间
freq_0 = 1000  # 频率0对应1000 Hz
freq_1 = 2000  # 频率1对应2000 Hz
preamble = '11111111'  # 前导码
max_payload_length = 96  # 最大负载长度（比特数）
preamble_bits = np.array([1 if b == '1' else -1 for b in preamble])  # 前导码二进制表示

# 实时音频接收设置
chunk_size = 1024  # 每次读取1024个样本
record_seconds = 5  # 每次录制5秒音频
audio_format = pyaudio.paInt16  # 音频格式

# 读取实时音频数据
def read_audio_stream():
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=audio_format,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("开始录音...")

    # 从流中读取音频数据
    frames = []
    for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(np.frombuffer(data, dtype=np.int16))

    # 关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames)

# 计算信号的自相关，找到与前导码最匹配的开始点
def find_preamble_using_autocorrelation(signal, preamble):
    # 计算信号的自相关
    correlation = np.correlate(signal, preamble, mode='valid')
    start_idx = np.argmax(np.abs(correlation))  # 找到相关性最大的点
    return start_idx

# 提取信号中的频率成分（FSK解调）
def demodulate_fsk(signal):
    signal = np.array(signal)
    signal_length = len(signal)
    bit_count = int(signal_length / (sample_rate * bit_duration))
    decoded_bits = []

    for i in range(bit_count):
        start_idx = int(i * sample_rate * bit_duration)
        end_idx = int((i + 1) * sample_rate * bit_duration)
        segment = signal[start_idx:end_idx]

        # 计算频率
        freqs = fftfreq(len(segment), 1 / sample_rate)
        fft_vals = fft(segment)
        peak_freq = freqs[np.argmax(np.abs(fft_vals))]

        # 判断频率属于0还是1
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

# 完整的实时解码流程
def decode_signal_real_time():
    decoded_payloads = {}

    while True:
        # 读取音频数据流
        signal = read_audio_stream()

        # 查找前导码的起始位置
        start_idx = find_preamble_using_autocorrelation(signal, preamble_bits)
        if start_idx == -1:
            print("未找到前导码，等待下一次接收...")
            continue

        # 从同步点开始解调
        signal = signal[start_idx:]
        signal_bits = demodulate_fsk(signal)

        # 解码数据包
        preamble_idx = signal_bits.find(preamble)
        while preamble_idx != -1:
            # 获取包头的长度信息（8 bit）和包序号信息（8 bit）
            payload_length = int(signal_bits[preamble_idx + 8:preamble_idx + 16], 2)
            packet_rank = int(signal_bits[preamble_idx + 16:preamble_idx + 24], 2)
            print(f"包{packet_rank} 数据长度: {payload_length} bits")

            # 获取数据段（Payload）并将其添加到payload字典
            payload = signal_bits[preamble_idx + 24:preamble_idx + 24 + payload_length]
            decoded_payloads[packet_rank] = payload

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

# 执行实时解码
decode_signal_real_time()
