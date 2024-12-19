import numpy as np
import sounddevice as sd


# 1. 将文本转换为二进制比特串
def text_to_binary(input_text):
    """将文本转换为二进制比特串"""
    return ''.join(format(ord(c), '08b') for c in input_text)


# 2. 二进制比特串转QAM符号
def binary_to_qam(binary_data, M):
    """将二进制数据转换为QAM符号。M为QAM的阶数，例如QPSK为4，16-QAM为16"""
    symbols = []
    bits_per_symbol = int(np.log2(M))  # 每个符号的比特数
    for i in range(0, len(binary_data), bits_per_symbol):
        bits = binary_data[i:i + bits_per_symbol]
        if len(bits) < bits_per_symbol:  # 对于不够一个符号的情况
            bits = bits.ljust(bits_per_symbol, '0')
        # 将比特序列转换为十进制
        decimal_value = int(bits, 2)
        symbols.append(decimal_value)

    # 对符号进行QAM调制，假设每个符号都是复数的 (I, Q)
    return np.array(symbols)  # 返回符号数组


# 3. 生成声波信号
def generate_sound_wave(qam_symbols, fs, f0, symbol_duration):
    """生成基于QAM符号的调制信号。`fs`为采样频率，`f0`为基准频率，`symbol_duration`为每个符号的持续时间"""
    t = np.linspace(0, symbol_duration, int(fs * symbol_duration), endpoint=False)  # 时间轴
    sound_wave = []

    for symbol in qam_symbols:
        carrier_freq = f0 + symbol * 1000  # 假设每个符号对应不同的频率
        carrier_signal = np.cos(2 * np.pi * carrier_freq * t)  # 生成载波信号
        sound_wave.extend(carrier_signal)

    return np.array(sound_wave)


# 4. 添加循环前缀（Cyclic Prefix）到每个符号
def add_cyclic_prefix(signal, cp_length, fs, symbol_duration):
    """为调制信号的每个符号添加循环前缀"""
    num_samples_cp = int(cp_length * fs * symbol_duration)  # 计算循环前缀的样本数
    signal_with_cp = []
    for i in range(0, len(signal), int(fs * symbol_duration)):  # 按照每个符号进行处理
        symbol = signal[i:i + int(fs * symbol_duration)]
        cp_signal = symbol[-num_samples_cp:]  # 从符号的末尾取出循环前缀部分
        signal_with_cp.extend(cp_signal)
        signal_with_cp.extend(symbol)  # 添加循环前缀后的符号
    return np.array(signal_with_cp)


# 5. 添加保护间隔
def add_guard_interval(signal, guard_interval_length):
    """为信号添加保护间隔"""
    guard_interval = np.zeros(guard_interval_length)  # 生成保护间隔（为零的样本）
    return np.concatenate([guard_interval, signal])  # 在信号前添加保护间隔


# 6. 播放生成的声波信号
def play_sound(sound_wave, fs):
    """播放生成的声波信号"""
    sd.play(sound_wave, fs)
    sd.wait()  # 等待声音播放完成


# 7. 创建数据包结构（前导码、包头和数据内容）
def create_data_packet(binary_data, packet_id, total_packets):
    """创建一个数据包，包括前导码（Preamble）、包头（Header）和数据内容段（Payload）"""
    preamble = '11111111'  # 前导码（8位）
    payload_max_size = 96  # 每个数据包的最大长度为96位
    payload = binary_data[:payload_max_size]  # 截取数据内容段（Payload）

    # 包头：长度、排名和总长度
    payload_length = format(len(payload), '08b')  # 数据内容长度（8位）
    packet_rank = format(packet_id, '08b')  # 数据包的排名（8位）
    total_packet_length = format(len(binary_data), '08b')  # 总数据包长度（8位）

    # 拼接数据包
    header = payload_length + packet_rank + total_packet_length
    data_packet = preamble + header + payload

    return data_packet


# 8. 将长文本拆分为多个数据包
def split_text_into_packets(binary_data, packet_size):
    """将长文本分割为多个数据包，每个数据包的大小为packet_size"""
    packets = []
    num_packets = len(binary_data) // packet_size + (1 if len(binary_data) % packet_size else 0)

    for i in range(num_packets):
        start_idx = i * packet_size
        end_idx = min((i + 1) * packet_size, len(binary_data))
        packets.append(binary_data[start_idx:end_idx])

    return packets


# 主函数
def main():
    # 用户输入文本
    input_text = input("请输入文本：")

    # 1. 文本转二进制
    binary_data = text_to_binary(input_text)
    print("二进制数据：", binary_data)

    # 2. 将二进制数据拆分成多个数据包
    packet_size = 96  # 每个数据包的最大大小（比特数）
    packets = split_text_into_packets(binary_data, packet_size)
    print("拆分后的数据包：", packets)

    # 3. 创建数据包并调制QAM符号
    M = 4  # QPSK
    fs = 44100  # 采样频率（Hz）
    f0 = 1000  # 基准频率（Hz）
    symbol_duration = 0.1  # 每个符号的持续时间（秒）

    # 循环前缀长度（符号数）
    cp_length = 4  # 循环前缀的符号数

    # 保护间隔长度（样本数）
    guard_interval_length = 1000  # 保护间隔的长度（样本数）

    # 4. 对每个数据包进行处理
    for i, packet in enumerate(packets):
        data_packet = create_data_packet(packet, i + 1, len(packets))  # 创建数据包
        print(f"数据包 {i + 1}：", data_packet)

        # 将数据包的二进制内容转为QAM符号
        qam_symbols = binary_to_qam(data_packet, M)
        print(f"QAM符号 {i + 1}：", qam_symbols)

        # 生成并播放声波信号
        sound_wave = generate_sound_wave(qam_symbols, fs, f0, symbol_duration)

        # 添加循环前缀
        sound_wave_with_cp = add_cyclic_prefix(sound_wave, cp_length, fs, symbol_duration)

        # 添加保护间隔
        sound_wave_with_guard_interval = add_guard_interval(sound_wave_with_cp, guard_interval_length)

        print(f"播放数据包 {i + 1}")
        play_sound(sound_wave_with_guard_interval, fs)


if __name__ == "__main__":
    main()
