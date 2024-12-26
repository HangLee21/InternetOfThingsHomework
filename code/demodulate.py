import os
import tkinter as tk
import pyaudio
import scipy
from scipy.io import wavfile
from scipy.io.wavfile import read
import numpy as np

from utils import rsdecode, binarray2barray

time_interval = None
PREAMBLE_WINDOW_SIZE = None
DATA_WINDOW_SIZE = None

# 解调参数
sample_rate = 48000  # 采样率
bit_duration = 0.1  # 每个比特的持续时间
freq_0 = 3750  # 频率0对应1000 Hz
freq_1 = 7500  # 频率1对应2000 Hz
PACKET_LENGTH = 12
CRC_LENGTH = 4
PREAMBLE = [1, 1, 1, 1, 1, 1, 1, 1]


# 读取声波信号并提取数据
def read_wave(file_path):
    rate, data = read(file_path)
    if len(data.shape) == 2:  # 如果是立体声，将其转换为单声道
        data = data.mean(axis=1)
    return rate, data


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
        assert len(data) == (PACKET_LENGTH + CRC_LENGTH) * 8
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
    data_symbol_num = (PACKET_LENGTH + CRC_LENGTH) * 8

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
    return start + int(bit_duration * n / time_interval)


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


# 拼接多个数据包的负载内容
def concatenate_payloads(payloads):
    return ''.join(payloads)


def demodulate_signal_wav(file_path):
    _, audio_sequence = wavfile.read(str(file_path))
    data_packets = fr_demodulate(audio_sequence)
    print(data_packets)
    result = ''
    for i, data in enumerate(data_packets):
        data_string = rsdecode(binarray2barray(data))
        result += data_string
    result = result.replace('\0', '')
    print(result)
    return result


if __name__ == '__main__':
    demodulate_signal_wav("output/record.wav")
