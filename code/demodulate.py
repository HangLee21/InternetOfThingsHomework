import scipy
from scipy.io import wavfile
from scipy.io.wavfile import read
import numpy as np
from scipy.signal import istft

from utils import rsdecode, binarray2barray

# 解调相关参数
SAMPLE_FREQUENCY = 48000  # 采样频率
SYMBOL_DURATION = 0.05  # 每个符号的持续时间
FREQ_LOW = 3750
FREQ_HIGH = 7500
DATA_PACKET_SIZE = 12  # 每个数据包的符号数
CHECKSUM_SIZE = 4  # 校验位长度
PREAMBLE = [1, 1, 1, 1, 1, 1, 1, 1]  # 前导码

# 全局变量
TIME_DELTA = None
SYNC_WINDOW_SIZE = None
DATA_WINDOW_SIZE = None


# 读取音频文件并提取数据
def load_audio(file_path):
    """从WAV文件中读取音频并返回采样率及数据"""
    rate, data = read(file_path)
    if len(data.shape) == 2:  # 如果是立体声，转换为单声道
        data = data.mean(axis=1)
    return rate, data


def perform_demodulation(signal):
    """对输入信号进行调制解调"""
    global TIME_DELTA, SYNC_WINDOW_SIZE, DATA_WINDOW_SIZE

    # 计算信号的频谱
    freq, time, spectrogram = scipy.signal.spectrogram(signal, SAMPLE_FREQUENCY, nperseg=256)
    spectrogram = np.abs(spectrogram)

    # 计算时间间隔和窗口大小
    TIME_DELTA = time[1] - time[0]
    SYNC_WINDOW_SIZE = int(SYMBOL_DURATION / TIME_DELTA)
    DATA_WINDOW_SIZE = int(SYMBOL_DURATION / TIME_DELTA)

    current_time_index = 0
    decoded_packets = []

    while current_time_index < len(time):
        # 查找同步前导码
        found_sync, sync_end, sync_data = locate_preamble(freq, time, spectrogram, current_time_index)
        if not found_sync:
            break  # 如果未找到同步前导码，则停止解调

        data_start_time = sync_end
        packet_data, next_data_end_time, raw_signals = decode_packet_data(freq, time, spectrogram,
                                                                          data_start_time)
        decoded_packets.append(packet_data)
        current_time_index = next_data_end_time  # 更新当前解调时间

    return decoded_packets


def normalize_amplitude(signal, start_idx, num_symbols):
    """对信号进行归一化处理"""
    return signal / np.max(signal[start_idx:start_idx + int(SYMBOL_DURATION * num_symbols / TIME_DELTA)])


def decode_packet_data(freq, time, spectrogram, start_time_index):
    """解码数据部分"""
    low_freq_idx, high_freq_idx, filtered_signal_low, filtered_signal_high, raw_signal_low, raw_signal_high = signal_prepare(freq, spectrogram)

    decoded_bits = []  # 存储解调后的数据
    total_symbols = (DATA_PACKET_SIZE + CHECKSUM_SIZE) * 8  # 总符号数量

    # 归一化信号
    raw_signal_low = normalize_amplitude(raw_signal_low, start_time_index, total_symbols)
    raw_signal_high = normalize_amplitude(raw_signal_high, start_time_index, total_symbols)

    time_idx = start_time_index  # 初始化时间索引
    symbol_count = 0  # 初始化符号计数

    while time_idx < len(filtered_signal_high) and symbol_count < total_symbols:
        # 选择信号段
        high_segment = raw_signal_high[time_idx: time_idx + DATA_WINDOW_SIZE]
        low_segment = raw_signal_low[time_idx: time_idx + DATA_WINDOW_SIZE]

        comparison_result = np.where(high_segment > low_segment, 1, 0)
        if np.average(comparison_result) < 0.5:
            decoded_bits.append(0)
        else:
            decoded_bits.append(1)

        symbol_count += 1
        time_idx = to_data_position(start_time_index, symbol_count)  # 更新时间索引


    return decoded_bits, time_idx, (filtered_signal_low, filtered_signal_high, time)

    # while time_idx < len(filtered_signal_high) and symbol_count < total_symbols:
    #     # 选择信号段
    #     high_segment = raw_signal_high[time_idx: time_idx + DATA_WINDOW_SIZE]
    #     low_segment = raw_signal_low[time_idx: time_idx + DATA_WINDOW_SIZE]
    #
    #     # 计算平均值
    #     avg_high = np.mean(high_segment)
    #     avg_low = np.mean(low_segment)
    #
    #     # 通过比较平均值来解码比特
    #     if avg_low > avg_high:
    #         decoded_bits.append(0)
    #     else:
    #         decoded_bits.append(1)
    #
    #     symbol_count += 1
    #     time_idx = to_data_position(start_time_index, symbol_count)  # 更新时间索引
    #
    # return decoded_bits, time_idx, (filtered_signal_low, filtered_signal_high, time)

def to_data_position(start, count):
    """计算数据段的位置"""
    return start + int(SYMBOL_DURATION * count / TIME_DELTA)


def get_frequency_index(freq, target_freq):
    """返回目标频率的索引"""
    return np.argmin(np.abs(freq - target_freq))


def compute_convolution(idx, signal):
    """计算信号的卷积值"""
    conv = 2 * signal[idx] - signal[idx - 1] - signal[idx + 1]
    return np.where(conv > 0, conv, 1e-7)


def apply_frequency_filter(freq_idx, signal):
    """对信号应用频率滤波"""
    filtered_band = compute_convolution(freq_idx, signal)
    filtered_band_lower = compute_convolution(freq_idx - 1, signal)
    filtered_band_upper = compute_convolution(freq_idx + 1, signal)
    diff_lower = np.log10(filtered_band) - np.log10(filtered_band_lower)
    diff_upper = np.log10(filtered_band) - np.log10(filtered_band_upper)

    near_comp = np.multiply(np.where(diff_lower > 5, 1, 0),
                            np.where(diff_upper > 5, 1, 0))
    self_comp = np.where(filtered_band > np.median(filtered_band), 1, 0)
    return np.multiply(near_comp, self_comp)


def bandpass_filter(freq, Sxx, center_freq, bandwidth):
    """
    过滤出指定频率周围的信号

    参数:
    signal: 输入信号
    sample_frequency: 采样频率
    center_freq: 中心频率
    bandwidth: 带宽（过滤范围的一半）

    返回:
    filtered_signal: 滤波后的信号
    freq: 频率数组
    time: 时间数组
    Sxx: 滤波后的频谱
    """

    # 创建频率掩码
    freq_mask = (freq >= (center_freq - bandwidth)) & (freq <= (center_freq + bandwidth))

    # 创建滤波后的频谱
    filtered_Sxx = np.zeros_like(Sxx)
    filtered_Sxx[freq_mask, :] = Sxx[freq_mask, :]

    # 使用逆短时傅里叶变换（ISTFT）将滤波后的谱图转换回时域信号
    _, filtered_signal = istft(filtered_Sxx, fs=SAMPLE_FREQUENCY, nperseg=256)

    return filtered_signal

def compute_symbol_confidence(raw_signal, time_idx, window_size):
    """计算符号的置信度"""
    next_time_idx = time_idx + window_size
    confidence = np.sum(raw_signal[time_idx: next_time_idx]) / window_size
    return confidence


def verify_sync_symbol_start(raw_signal_low, raw_signal_high, time_idx, symbol=1):
    """验证同步符号的开始位置"""
    confidence_low = compute_symbol_confidence(raw_signal_low, time_idx, SYNC_WINDOW_SIZE)
    confidence_high = compute_symbol_confidence(raw_signal_high, time_idx, SYNC_WINDOW_SIZE)

    if symbol == 0:
        return confidence_low > 0.01 * np.max(raw_signal_low) and np.log10(confidence_low) - np.log10(
            confidence_high) > 0.2
    else:
        return confidence_high > 0.01 * np.max(raw_signal_high) and np.log10(confidence_high) - np.log10(
            confidence_low) > 0.2


def find_next_symbol(filtered_signal, current_idx, symbol=1):
    """查找下一个符号的位置"""
    if filtered_signal[current_idx] == symbol:
        if current_idx + 1 >= len(filtered_signal):
            return len(filtered_signal)
        res = current_idx + 1 + np.argmax(filtered_signal[current_idx + 1:] == symbol)
        if filtered_signal[res] != symbol:
            return len(filtered_signal)
        return res
    else:
        res = current_idx + np.argmax(filtered_signal[current_idx:] == symbol)
        if res == current_idx:
            return len(filtered_signal)
        return res

def signal_prepare(freq, signal_spectrogram):
    low_freq_idx = get_frequency_index(freq, FREQ_LOW)
    high_freq_idx = get_frequency_index(freq, FREQ_HIGH)
    filtered_signal_low = apply_frequency_filter(low_freq_idx, signal_spectrogram)
    filtered_signal_high = apply_frequency_filter(high_freq_idx, signal_spectrogram)
    raw_signal_low = compute_convolution(low_freq_idx, signal_spectrogram)
    raw_signal_high = compute_convolution(high_freq_idx, signal_spectrogram)

    return low_freq_idx, high_freq_idx, filtered_signal_low, filtered_signal_high, raw_signal_low, raw_signal_high

def locate_preamble(freq, time, signal_spectrogram, start_index):
    """定位同步前导码的位置"""
    low_freq_idx, high_freq_idx, filtered_signal_low, filtered_signal_high, raw_signal_low, raw_signal_high = signal_prepare(freq, signal_spectrogram)

    sync_start = start_index
    correct_length = 0

    while sync_start < len(filtered_signal_high):
        # 过滤不符合的信号
        if not verify_sync_symbol_start(raw_signal_low, raw_signal_high, sync_start, 1):
            sync_start = find_next_symbol(filtered_signal_high, sync_start)
            if sync_start >= len(filtered_signal_high):
                print('查找完成')
                break

        # 校验同步模式
        while correct_length < len(PREAMBLE):
            if verify_sync_symbol_start(raw_signal_low, raw_signal_high, sync_start, 1):
                correct_length += 1
                sync_start += SYNC_WINDOW_SIZE
                if correct_length == len(PREAMBLE):
                    break
            else:
                correct_length = 0
                break

        if correct_length == len(PREAMBLE):
            return True, sync_start, (filtered_signal_low, filtered_signal_high, raw_signal_low, raw_signal_high, time)

    return False, len(filtered_signal_low), (
        filtered_signal_low, filtered_signal_high, raw_signal_low, raw_signal_high, time)


def demodulate_audio(file_path):
    """解调WAV文件中的音频信号"""
    _, audio_signal = wavfile.read(str(file_path))
    decoded_packets = perform_demodulation(audio_signal)  # 解调信号
    print(decoded_packets)

    final_result = ''
    binary_packets = []
    for i, packet in enumerate(decoded_packets):
        # 进行纠错解码
        decoded_string = rsdecode(binarray2barray(packet))
        final_result += decoded_string
        binary_packets.append(packet)

    binary_array = [item for sublist in binary_packets for item in sublist]
    final_result = final_result.replace('\0', '')  # 去除多余的空字符
    print(final_result)
    return final_result, binary_array


if __name__ == '__main__':
    demodulate_audio("output/record.wav")
