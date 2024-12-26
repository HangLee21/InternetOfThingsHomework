import struct
import wave

import numpy as np
from functools import reduce

import pyaudio

from utils import rsencode, barray2binarray

# 调制参数
sample_rate = 48000  # 采样率
bit_duration = 0.1  # 每个比特的持续时间
freq_0 = 3750  # 频率0对应1000 Hz
freq_1 = 7500  # 频率1对应2000 Hz
PACKET_TEXT_LENGTH = 12
PREAMBLE = [1, 1, 1, 1, 1, 1, 1, 1]
GAP_INTERVAL = 0.2
AMPLITUDE = 5000
START_INTERVAL = 1
"""
数据包结构:
preamble: 11111111
payload: 最大长度 = 96 bit
CRC: 32 bit
"""


# 汉明码编码器（7,4编码）
def make_01_raw_seq(text):
    raw_seq = np.array([]).astype(np.int32)
    text_split = [text[i:i + PACKET_TEXT_LENGTH] for i in range(0, len(text), PACKET_TEXT_LENGTH)]
    for i in range(len(text_split)):
        if len(text_split[i]) < PACKET_TEXT_LENGTH:
            text_split[i] = text_split[i] + '\0' * (PACKET_TEXT_LENGTH - len(text_split[i]))
        rs_data = rsencode(text_split[i])
        raw_seq = np.append(raw_seq, barray2binarray(rs_data))
    return raw_seq


def make_send_packets(text):
    packets = []
    text_split = [text[i:i + PACKET_TEXT_LENGTH] for i in range(0, len(text), PACKET_TEXT_LENGTH)]
    start_packet = fr_modulate_freq(0, START_INTERVAL)

    packets.append(start_packet)
    for i in range(len(text_split)):
        packet = np.array([])

        preamble_packet = fr_modulate(PREAMBLE, bit_duration, sig_0_freq=freq_0, sig_1_freq=freq_1)
        packet = np.append(packet, preamble_packet)

        if len(text_split[i]) < PACKET_TEXT_LENGTH:
            text_split[i] = text_split[i] + '\0' * (PACKET_TEXT_LENGTH - len(text_split[i]))

        rs_data = rsencode(text_split[i])
        rs_data_packet = fr_modulate(barray2binarray(rs_data), bit_duration, sig_0_freq=freq_0, sig_1_freq=freq_1)
        packet = np.append(packet, rs_data_packet)

        gap_packet = fr_modulate_freq(0, GAP_INTERVAL)
        packet = np.append(packet, gap_packet)

        packets.append(packet)

    result = reduce(lambda x, y: np.append(x, y), packets)
    return result


# 主程序
def main():
    # 控制是否存储为 wav 文件或直接播放声音
    save_to_wav = True  # 设置为 True 来保存为 wav 文件，设置为 False 来播放声音

    text = input("请输入文本：")

    signals = make_send_packets(text)

    save_wave(signals)

    play_wav('output/record.wav')


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


def save_wave(signal):
    wf = wave.open(str('output/record.wav'), 'wb')
    wf.setnchannels(1)
    wf.setframerate(sample_rate)
    wf.setsampwidth(2)
    for i in range(len(signal)):
        the_result = int(AMPLITUDE * signal[i])
        data = struct.pack('<h', the_result)
        wf.writeframesraw(data)
    wf.close()

def play_wav(file_path):
    # 打开WAV文件
    wf = wave.open(file_path, 'rb')

    # 创建PyAudio对象
    p = pyaudio.PyAudio()

    # 打开流
    stream = p.open(format=pyaudio.paInt16,
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 读取WAV文件数据
    data = wf.readframes(1024)

    # 播放WAV文件
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # 关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

# 运行主程序
main()

