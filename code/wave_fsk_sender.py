import struct
import wave
import numpy as np
from functools import reduce
import pyaudio

from utils import rsencode, barray2binarray

# 调制参数
SAMPLE_RATE = 48000  # 采样率
BIT_DURATION = 0.1  # 每个比特的持续时间
FREQ_0 = 3750  # 频率0对应1000 Hz
FREQ_1 = 7500  # 频率1对应2000 Hz
PACKET_TEXT_LENGTH = 12
PREAMBLE = [1, 1, 1, 1, 1, 1, 1, 1]
GAP_INTERVAL = 0.2
AMPLITUDE = 5000
START_INTERVAL = 1


# 数据包结构:
# preamble: 11111111
# payload: 最大长度 = 96 bit
# CRC: 32 bit


def modulate_signal(code_seq, sym_dur, sig_0_freq=FREQ_0, sig_1_freq=FREQ_1):
    """对比特序列进行调制"""
    N = int(SAMPLE_RATE * sym_dur)  # 每个比特的采样点数
    t = np.arange(N) / SAMPLE_RATE  # 时间轴

    # 基本信号
    base_signal_0 = np.sin(2 * np.pi * sig_0_freq * t)
    base_signal_1 = np.sin(2 * np.pi * sig_1_freq * t)

    # 调制信号：根据比特选择信号
    modulated_signal = np.zeros(N * len(code_seq))
    for i, bit in enumerate(code_seq):
        modulated_signal[i * N:(i + 1) * N] = base_signal_0 if bit == 0 else base_signal_1

    return modulated_signal


def make_send_packets(text):
    """根据输入文本生成完整的发送数据包"""
    packets = []

    # 将文本拆分为多个包
    text_split = [text[i:i + PACKET_TEXT_LENGTH] for i in range(0, len(text), PACKET_TEXT_LENGTH)]

    # 添加启动信号包
    start_packet = modulate_signal([0], START_INTERVAL)
    packets.append(start_packet)

    # 遍历每个数据包
    for chunk in text_split:
        packet = np.array([])

        # 添加前导码
        preamble_packet = modulate_signal(PREAMBLE, BIT_DURATION, sig_0_freq=FREQ_0, sig_1_freq=FREQ_1)
        packet = np.append(packet, preamble_packet)

        # 填充并调制数据
        if len(chunk) < PACKET_TEXT_LENGTH:
            chunk = chunk + '\0' * (PACKET_TEXT_LENGTH - len(chunk))  # 填充至固定长度
        rs_data = rsencode(chunk)
        rs_data_packet = modulate_signal(barray2binarray(rs_data), BIT_DURATION, sig_0_freq=FREQ_0, sig_1_freq=FREQ_1)
        packet = np.append(packet, rs_data_packet)

        # 添加间隔信号
        gap_packet = modulate_signal([0], GAP_INTERVAL)
        packet = np.append(packet, gap_packet)

        packets.append(packet)

    # 合并所有包并返回
    return reduce(lambda x, y: np.append(x, y), packets)


def save_wave(signal, file_path='output/record.wav'):
    """将调制信号保存为WAV文件"""
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setframerate(SAMPLE_RATE)  # 设置采样率
        wf.setsampwidth(2)  # 设置每个采样点的字节数

        # 写入信号数据
        for sample in signal:
            wf.writeframesraw(struct.pack('<h', int(AMPLITUDE * sample)))


def play_wav(file_path):
    """播放WAV文件"""
    with wave.open(file_path, 'rb') as wf:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # 播放数据
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        # 关闭流
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    """主程序"""
    text = input("请输入文本：")

    # 生成发送数据包
    signals = make_send_packets(text)
    save_wave(signals)
    print("信号已保存为 WAV 文件。")
    play_wav('output/record.wav')



if __name__ == '__main__':
    main()
