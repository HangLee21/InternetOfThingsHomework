import os
import numpy as np
import tkinter as tk
import pyaudio
from scipy.io.wavfile import read
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

# 解调参数
sample_rate = 44100  # 采样率
bit_duration = 0.1  # 每个比特的持续时间
freq_0 = 1000  # 频率0对应1000 Hz
freq_1 = 2000  # 频率1对应2000 Hz
max_payload_length = 192  # 最大负载长度（比特数）

# 带通滤波器设计
def bandpass_filter(signal, lowcut, highcut, sample_rate):
    nyquist = 0.5 * sample_rate  # 奈奎斯特频率
    low = lowcut / nyquist  # 低频截止频率
    high = highcut / nyquist  # 高频截止频率
    b, a = butter(4, [low, high], btype='band')  # 4阶滤波器
    filtered_signal = filtfilt(b, a, signal)  # 应用滤波器
    return filtered_signal


# 读取声波信号并提取数据
def read_wave(file_path):
    rate, data = read(file_path)
    if len(data.shape) == 2:  # 如果是立体声，将其转换为单声道
        data = data.mean(axis=1)
    return rate, data


# 缓冲区，用于缓存接收到的音频信号
buffer = []

# 读取实时音频流
def record_audio_stream():
    global buffer
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
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
                signal_bits = demodulate_fsk(buffer)  # 调用解调函数
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

                    # 等待足够的header和payload数据
                    header_end_idx = preamble_idx + len(preamble) + 24  # header为24位
                    if len(signal_bits) >= header_end_idx:
                        header_bits = signal_bits[preamble_idx + len(preamble):header_end_idx]
                        # 解析header
                        payload_length, packet_rank, total_packet_length = parse_header(header_bits)
                        total_packets = total_packet_length  # 获取总包数

                        if len(signal_bits) >= 24:
                            signal = np.array(buffer) - np.mean(buffer)

                            # 计算每个比特的FFT
                            signal_length = len(signal)
                            bit_count = int(signal_length / (sample_rate * bit_duration))  # 每个比特的时长

                            for i in range(bit_count):
                                if i > preamble_idx - 2:
                                    start_idx = int(i * sample_rate * bit_duration)
                                    end_idx = int((i + 1) * sample_rate * bit_duration)
                                    segment = signal[start_idx:end_idx]

                                    # Perform FFT on the current signal segment
                                    freqs = np.fft.fftfreq(len(segment), 1 / sample_rate)
                                    fft_vals = np.fft.fft(segment)
                                    fft_mag = np.abs(fft_vals)

                                    # Calculate the peak frequency (find the index with the maximum magnitude)
                                    peak_freq = freqs[np.argmax(fft_mag)]

                                    # 绘制每个比特的FFT图
                                    plt.figure(figsize=(10, 6))
                                    plt.plot(freqs[:len(freqs) // 2], fft_mag[:len(fft_mag) // 2])
                                    plt.title(f"FFT of Bit {i + 1} (Peak Frequency: {abs(peak_freq)} Hz)")
                                    plt.xlabel("Frequency (Hz)")
                                    plt.ylabel("Magnitude")
                                    plt.grid(True)
                                    plt.show()

                                    print(f'当前位置: {i}, 波峰频率: {abs(peak_freq)}')

                            break

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

# 解析header信息（24位）
def parse_header(header_bits):
    payload_length = int(header_bits[0:8], 2)  # 前8位是payload长度
    packet_rank = int(header_bits[8:16], 2)  # 接下来的8位是包序号
    total_packet_length = int(header_bits[16:24], 2)  # 后8位是总包数
    print(f"包序号: {packet_rank}, payload 长度: {payload_length} bits, 总包数: {total_packet_length}")
    return payload_length, packet_rank, total_packet_length


# 解码数据包
def decode_data_packet(payload_bits):
    # 解码payload（假设payload不含错误，需要汉明解码）
    decoded_payload = hamming_decode(payload_bits)  # 纠错后得到有效负载
    return binary_to_text(decoded_payload)



# 提取信号中的频率成分（FSK解调）
def demodulate_fsk(signal):
    # 使用滤波后的信号
    signal = np.array(signal) - np.mean(signal)

    # 计算每个比特的FFT
    signal_length = len(signal)
    bit_count = int(signal_length / (sample_rate * bit_duration))  # 每个比特的时长
    decoded_bits = []



    cnt = 0
    # 遍历信号，找到前导码的位置
    for i in range(bit_count):
        start_idx = int(i * sample_rate * bit_duration)
        end_idx = int((i + 1) * sample_rate * bit_duration)
        segment = signal[start_idx:end_idx]

        # Perform FFT on the current signal segment
        freqs = np.fft.fftfreq(len(segment), 1 / sample_rate)
        fft_vals = np.fft.fft(segment)
        fft_mag = np.abs(fft_vals)

        # Set the FFT result to zero for frequencies below 500 Hz
        mask = np.abs(freqs) < 500
        fft_mag[mask] = 0  # Zero out frequencies lower than 500 Hz
        fft_vals[mask] = 0  # Zero out the corresponding FFT values as well

        # Calculate the peak frequency (find the index with the maximum magnitude)
        peak_freq = freqs[np.argmax(fft_mag)]

        tolerance = 100
        # Check if the peak frequency corresponds to 0 or 1
        if abs(abs(peak_freq) - freq_0) < tolerance:
            decoded_bits.append('0')
        elif abs(abs(peak_freq) - freq_1) < tolerance:
            decoded_bits.append('1')
        else:
            decoded_bits.append('2')

    signal_bits = ''.join(decoded_bits)
    return signal_bits



# 解码数据包
def decode_data_packet(signal_bits, preamble_idx):
    # 获取包头的长度信息（8 bit）和包序号信息（8 bit）
    payload_length = int(signal_bits[preamble_idx + 8:preamble_idx + 16], 2)
    packet_rank = int(signal_bits[preamble_idx + 16:preamble_idx + 24], 2)  # 解析包序号
    total_packet_length = int(signal_bits[preamble_idx + 24:preamble_idx + 32], 2)  # 解析总包数
    print(f"包{packet_rank} 数据长度: {payload_length} bits, 总包数: {total_packet_length}")

    # 获取数据段（Payload），并加入汉明解码
    payload = signal_bits[preamble_idx + 32:preamble_idx + 32 + payload_length]
    corrected_payload = hamming_decode(payload)  # 纠错后得到有效负载

    # 如果接收到的包是最后一个包，则停止监听
    if packet_rank == total_packet_length:
        print("接收到最后一个包，结束接收。")

    return binary_to_text(corrected_payload), packet_rank  # 返回有效负载


# 汉明码解码器（纠错）
def hamming_decode(bits):
    """ 对输入的比特串进行汉明码解码，返回纠错后的有效比特 """
    # 汉明(7, 4)编码，7位中4位是数据位，3位是校验位
    n = len(bits)
    corrected_bits = []
    for i in range(0, n, 7):
        # 每7位为一组
        block = bits[i:i + 7]
        if len(block) < 7:
            break
        # 校验位位置：0、1、3
        p1 = block[0]
        p2 = block[1]
        p3 = block[3]
        d1 = block[2]
        d2 = block[4]
        d3 = block[5]
        d4 = block[6]

        # 计算校验位的值
        parity_1 = int(p1) ^ int(d1) ^ int(d2) ^ int(d4)
        parity_2 = int(p2) ^ int(d1) ^ int(d3) ^ int(d4)
        parity_3 = int(p3) ^ int(d2) ^ int(d3) ^ int(d4)

        # 检查错误并纠正
        error_pos = parity_1 * 1 + parity_2 * 2 + parity_3 * 4
        if error_pos != 0:
            print(f"检测到错误，位置：{error_pos}, 正在纠正")
            block[error_pos - 1] = '1' if block[error_pos - 1] == '0' else '0'

        # 提取数据位
        corrected_bits.extend([block[2], block[4], block[5], block[6]])

    return ''.join(corrected_bits)


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
def decode_signal(file_paths=None, use_file_input=True):
    decoded_payloads = {}  # 用来存储包序号和解码的文本
    packet_rank = 1
    text = ''

    if use_file_input:
        for file_path in file_paths:
            # 读取音频文件
            rate, signal = read_wave(file_path)
            signal_bits = demodulate_fsk(signal)
            preamble = '11111111'  # 前导码
            preamble_idx = signal_bits.find(preamble)

            if preamble_idx != -1:
                # 解码数据包并返回解码后的文本和包序号
                decoded_text, packet_rank = decode_data_packet(signal_bits, preamble_idx)
                decoded_payloads[packet_rank] = decoded_text  # 将文本保存到字典中，键为包序号

        # 按照 packet_rank 排序字典的键
        sorted_payloads = sorted(decoded_payloads.items(), key=lambda x: x[0])  # 按包序号排序
        # 拼接排序后的文本
        full_text = ''.join([text for _, text in sorted_payloads])  # 拼接所有包的文本

        display_decoded_text(full_text)  # 显示完整的解码文本

    else:
        record_audio_stream()


# 示例：执行解码流程
use_file_input = False  # 设置为 False 来实时接收音频信号
if use_file_input:
    directory = 'output'
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
    decode_signal(file_paths=file_paths, use_file_input=True)
else:
    decode_signal(use_file_input=False)  # 实时接收音频
