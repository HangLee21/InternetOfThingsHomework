import os
import tkinter as tk
import pyaudio
import scipy
from scipy.io.wavfile import read
import numpy as np

from matplotlib import pyplot as plt

time_interval = None
PREAMBLE_WINDOW_SIZE = None
DATA_WINDOW_SIZE = None

# 解调参数
sample_rate = 44100  # 采样率
bit_duration = 0.1  # 每个比特的持续时间
freq_0 = 1000  # 频率0对应1000 Hz
freq_1 = 2000  # 频率1对应2000 Hz
max_payload_length = 192  # 最大负载长度（比特数）
packet_length = (max_payload_length + 24) / 8
CRC_LENGTH = 0
PREAMBLE = [1, 1, 1, 1, 1, 1, 1, 1]

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
    data_packets = []
    main_t_idx = 0
    try:
        while True:
            data = stream.read(1024)
            buffer.extend(np.frombuffer(data, dtype=np.int16))  # 将新数据添加到缓冲区

            # 检查是否有足够的数据来解码

            if len(buffer) > (sample_rate * bit_duration * (packet_length * 8 + 8) ) * 2 + main_t_idx:
                global time_interval, PREAMBLE_WINDOW_SIZE, DATA_WINDOW_SIZE
                f, t, Zxx = scipy.signal.spectrogram(buffer, sample_rate, nperseg=256)
                Zxx = np.abs(Zxx)

                time_interval = t[1] - t[0]
                PREAMBLE_WINDOW_SIZE = int(bit_duration / time_interval)
                DATA_WINDOW_SIZE = int(bit_duration / time_interval)

                have_preamble, preamble_end, signal_bits = find_preamble(f, t, Zxx, main_t_idx)

                if have_preamble:
                    print("检测到前导码，开始解码数据包...")
                    data_start = preamble_end
                    data, data_end, raw, logs = decode_data(f, t, Zxx, data_start)
                    assert len(data) == (packet_length + CRC_LENGTH) * 8

                    decimal_list, header, char_list = binary_to_decimal(data)
                    payload_len = header[0] / 8
                    payload_rank = header[1]
                    total_packets = header[2]
                    data_packets.append({
                        'payload_rank': payload_rank,
                        'char_list': char_list[:payload_len]
                    })
                    main_t_idx = data_end
                    total_packets_received = total_packets_received + 1

                    if total_packets_received == total_packets:
                        print('接收完成')
                        break
                else:
                    main_t_idx = main_t_idx +1
                    continue
    except KeyboardInterrupt:
        print("停止接收音频信号")
        stream.stop_stream()
        stream.close()
        p.terminate()


def binary_to_decimal(binary_data):
    # 确保 binary_data 是由 0 和 1 组成的列表
    if not all(bit in [0, 1] for bit in binary_data):
        raise ValueError("输入数据必须是由 0 和 1 组成的二进制列表")

    # 将二进制数据按每 8 位拆分为字节
    byte_list = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]

    # 将每个字节转换为十进制数字
    decimal_list = [int(''.join(map(str, byte)), 2) for byte in byte_list]

    # 提取前三个数字作为 header
    header = decimal_list[:3]

    # 将剩余的数字转换为字符（假设这些数字是 ASCII 码）
    char_list = [chr(num) for num in decimal_list[3:]]  # 从第四个数字开始转字符

    return decimal_list, header, char_list


def fr_modulate_freq(freq, t):
    N = int(sample_rate * t)
    x = np.arange(N) / sample_rate
    y = np.sin(2 * np.pi * freq * x)
    return y


def fr_modulate(code_seq, sym_dur, sig_0_freq=freq_0, sig_1_freq=freq_1):
    N = int(sample_rate * sym_dur)
    t = np.arange(N) / sample_rate
    base_signal_0 = np.sin(2 * np.pi * sig_0_freq * t)
    base_signal_1 = np.sin(2 * np.pi * sig_1_freq * t)
    modulated_signal = np.zeros((N * len(code_seq)))
    for i in range(len(code_seq)):
        modulated_signal[i * N: (i + 1) *
                                N] = base_signal_0 if code_seq[i] == 0 else base_signal_1
    return modulated_signal



def fr_demodulate(sig):
    global time_interval, PREAMBLE_WINDOW_SIZE, DATA_WINDOW_SIZE
    f, t, Zxx = scipy.signal.spectrogram(sig, sample_rate, nperseg=256)
    Zxx = np.abs(Zxx)

    time_interval = t[1] - t[0]
    PREAMBLE_WINDOW_SIZE = int(bit_duration / time_interval)
    DATA_WINDOW_SIZE = int(bit_duration / time_interval)

    main_t_idx = 0
    data_packets = []

    while main_t_idx < len(t):
        have_preamble, preamble_end, preamble_raw_data = find_preamble(f, t, Zxx, main_t_idx)
        if not have_preamble:
            break

        data_start = preamble_end
        data, data_end, raw, logs = decode_data(f, t, Zxx, data_start)
        assert len(data) == (packet_length + CRC_LENGTH) * 8
        data_packets.append(data)
        main_t_idx = data_end

    return data_packets


def decode_data(f, t, Zxx, data_start):
    f_0_idx = select_freq(f, freq_0)
    f_1_idx = select_freq(f, freq_1)
    filtered_sig_0 = filter_freq(f_0_idx, Zxx)
    filtered_sig_1 = filter_freq(f_1_idx, Zxx)
    raw_sig_0 = Zxx[f_0_idx]
    raw_sig_1 = Zxx[f_1_idx]

    logs = []
    data = []
    data_symbol_num = (packet_length + CRC_LENGTH) * 8

    raw_sig_0 = raw_sig_0 / np.max(
        raw_sig_0[data_start: data_start + to_data_segment_place(data_start, data_symbol_num)])
    raw_sig_1 = raw_sig_1 / np.max(
        raw_sig_1[data_start: data_start + to_data_segment_place(data_start, data_symbol_num)])

    t_idx = data_start
    n_symbol = 0

    t_idx = max(align_symbol(filtered_sig_0, t_idx, DATA_WINDOW_SIZE),
                align_symbol(filtered_sig_1, t_idx, DATA_WINDOW_SIZE))
    while t_idx < len(filtered_sig_1) and n_symbol < data_symbol_num:
        # 对齐
        # t_idx = max(align_symbol(filtered_sig_0, t_idx), align_symbol(filtered_sig_1, t_idx))
        if t_idx >= len(filtered_sig_1):
            break
        conf_0 = calculate_symbol_confidence(raw_sig_0, t_idx, DATA_WINDOW_SIZE)
        conf_1 = calculate_symbol_confidence(raw_sig_1, t_idx, DATA_WINDOW_SIZE)

        flag_1_over_0 = np.where(
            raw_sig_1[t_idx: t_idx + DATA_WINDOW_SIZE] > raw_sig_0[t_idx: t_idx + DATA_WINDOW_SIZE], 1, 0)

        if np.average(flag_1_over_0) < 0.5:
            data.append(0)
        else:
            data.append(1)

        logs.append({'t': t[to_data_segment_place(data_start, n_symbol)], 'conf_0': conf_0, 'conf_1': conf_1,
                     'symbol': data[-1]})
        n_symbol += 1
        t_idx = to_data_segment_place(data_start, n_symbol)

    return data, t_idx, (filtered_sig_0, filtered_sig_1, t), logs


def to_data_segment_place(start, n):
    return start + int( * n / time_interval)


def select_freq(f, target_f):
    """ 返回频率序列中最接近目标频率的序号 """
    return np.argmin(np.abs(f - target_f))


def calculate_conv(f_idx, sig_xx):
    assert f_idx > 0 and f_idx < sig_xx.shape[0] - 1
    conv = 2 * sig_xx[f_idx] - sig_xx[f_idx - 1] - sig_xx[f_idx + 1]
    return np.where(conv > 0, conv, 1e-7)


def filter_freq(f_idx, sig_xx):
    """ 分离指定的频率，返回一个01序列 """
    assert f_idx > 1 and f_idx < sig_xx.shape[0] - 2
    filtered_band = calculate_conv(f_idx, sig_xx)
    filtered_band_lower = calculate_conv(f_idx - 1, sig_xx)
    filtered_band_upper = calculate_conv(f_idx + 1, sig_xx)
    tmp_1 = np.log10(filtered_band) - np.log10(filtered_band_lower)
    tmp_2 = np.log10(filtered_band) - np.log10(filtered_band_upper)

    near_comp = np.multiply(np.where(tmp_1 > 8, 1, 0),
                            np.where(tmp_2 > 8, 1, 0))
    self_comp = np.where(filtered_band > np.median(filtered_band), 1, 0)
    return np.multiply(near_comp, self_comp)


def calculate_symbol_confidence(raw_sig, t_idx, winsz):
    """ 计算在指定频率序列的某一时间是一个symbol的开始的置信度 """
    """ t 是 index """
    t_next_idx = t_idx + winsz
    conf = np.sum(raw_sig[t_idx: t_next_idx]) / winsz
    return conf


def confirm_preamble_symbol_start(raw_sig_0, raw_sig_1, t_idx, t, symbol=1):
    """ 确认在指定频率序列的某一时间是否是一个前导码symbol的开始 """
    """ t 是 index """
    conf_0 = calculate_symbol_confidence(raw_sig_0, t_idx, PREAMBLE_WINDOW_SIZE)
    conf_1 = calculate_symbol_confidence(raw_sig_1, t_idx, PREAMBLE_WINDOW_SIZE)

    # print('\t{:.3f}: {:.3f}, {:.3f}'.format(t[t_idx], conf_0, conf_1))
    if symbol == 0:
        return conf_0 > 0.01 * np.max(raw_sig_0) and np.log10(conf_0) - np.log10(conf_1) > 0.2
    else:
        return conf_1 > 0.01 * np.max(raw_sig_1) and np.log10(conf_1) - np.log10(conf_0) > 0.2


def find_next_symbol_start(filtered_sig, current_idx, symbol=1):
    """ 寻找下一个 1 的开始，返回index，保证返回值一定大于current_idx，如果找不到，返回len(filtered_sig)"""
    if filtered_sig[current_idx] == symbol:
        if current_idx + 1 >= len(filtered_sig):
            return len(filtered_sig)
        res = current_idx + 1 + np.argmax(filtered_sig[current_idx + 1:] == symbol)
        if filtered_sig[res] != symbol:
            return len(filtered_sig)
        return res
    else:
        res = current_idx + np.argmax(filtered_sig[current_idx:] == symbol)
        if res == current_idx:
            return len(filtered_sig)
        return res


def align_symbol(filtered_sig, t_idx, winsz, type='start', symbol=1):
    ''' 对齐 t_idx 到一个 symbol 的开始，差距过大的话，不对齐，返回原来的 t_idx'''
    if type == 'start':
        if filtered_sig[t_idx] == symbol:
            # 这里容易死循环，先不写，因为int取整只有可能落后，不可能超前
            return t_idx
        else:
            offset = find_next_symbol_start(filtered_sig, t_idx) - t_idx
            if offset < winsz:
                return t_idx + offset
            return t_idx


def find_preamble(f, t, sig_xx, start):
    f_0_idx = select_freq(f, freq_0)
    f_1_idx = select_freq(f, freq_1)
    filtered_sig_0 = filter_freq(f_0_idx, sig_xx)
    filtered_sig_1 = filter_freq(f_1_idx, sig_xx)
    raw_sig_0 = calculate_conv(f_0_idx, sig_xx)
    raw_sig_1 = calculate_conv(f_1_idx, sig_xx)

    preamble_start = start
    correct_len = 0
    correct_target = PREAMBLE[correct_len]

    while preamble_start < len(filtered_sig_1):

        # find first symbol 1
        if not confirm_preamble_symbol_start(raw_sig_0, raw_sig_1, preamble_start, t, 1):
            preamble_start = find_next_symbol_start(filtered_sig_1, preamble_start)
            if preamble_start >= len(filtered_sig_1):
                print('finished')
                break

        # check preamble
        while correct_len < len(PREAMBLE):
            # print('correct_len: {}, current_time: {}, p_idx: {}'.format(correct_len, t[preamble_start], preamble_start))
            if correct_target == 1:
                # check 1
                preamble_start = align_symbol(filtered_sig_1, preamble_start, PREAMBLE_WINDOW_SIZE)
                if confirm_preamble_symbol_start(raw_sig_0, raw_sig_1, preamble_start, t, 1):
                    correct_len += 1
                    preamble_start += PREAMBLE_WINDOW_SIZE
                    if correct_len == len(PREAMBLE):
                        break
                    correct_target = PREAMBLE[correct_len]
                else:
                    correct_len = 0
                    correct_target = PREAMBLE[correct_len]
                    break
            else:
                # check 0
                preamble_start = align_symbol(filtered_sig_0, preamble_start, PREAMBLE_WINDOW_SIZE)
                if confirm_preamble_symbol_start(raw_sig_0, raw_sig_1, preamble_start, t, 0):
                    correct_len += 1
                    preamble_start += PREAMBLE_WINDOW_SIZE
                    if correct_len == len(PREAMBLE):
                        break
                    correct_target = PREAMBLE[correct_len]
                else:
                    correct_len = 0
                    correct_target = PREAMBLE[correct_len]
                    break

        if correct_len == len(PREAMBLE):
            print('find preamble ended at: {:.4f}s'.format(preamble_start * time_interval))
            return True, preamble_start, (filtered_sig_0, filtered_sig_1, raw_sig_0, raw_sig_1, t)

    return False, len(filtered_sig_0), (filtered_sig_0, filtered_sig_1, raw_sig_0, raw_sig_1, t)


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

    # if use_file_input:
    #     for file_path in file_paths:
    #         # 读取音频文件
    #         rate, signal = read_wave(file_path)
    #         signal_bits = demodulate_fsk(signal)
    #         preamble = '11111111'  # 前导码
    #         preamble_idx = signal_bits.find(preamble)
    #
    #         if preamble_idx != -1:
    #             # 解码数据包并返回解码后的文本和包序号
    #             decoded_text, packet_rank = decode_data_packet(signal_bits, preamble_idx)
    #             decoded_payloads[packet_rank] = decoded_text  # 将文本保存到字典中，键为包序号
    #
    #     # 按照 packet_rank 排序字典的键
    #     sorted_payloads = sorted(decoded_payloads.items(), key=lambda x: x[0])  # 按包序号排序
    #     # 拼接排序后的文本
    #     full_text = ''.join([text for _, text in sorted_payloads])  # 拼接所有包的文本
    #
    #     display_decoded_text(full_text)  # 显示完整的解码文本
    #
    # else:
    record_audio_stream()


# 示例：执行解码流程
use_file_input = False  # 设置为 False 来实时接收音频信号
if use_file_input:
    directory = 'output'
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
    decode_signal(file_paths=file_paths, use_file_input=True)
else:
    decode_signal(use_file_input=False)  # 实时接收音频
