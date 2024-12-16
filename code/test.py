import sounddevice as sd
import numpy as np

# 生成示例音频信号（例如，正弦波）
sample_rate = 44100  # 采样率
duration = 2  # 音频持续时间（秒）
freq = 440  # 音频频率（Hz）

t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * freq * t)  # 正弦波信号

# 将信号缩放为 [-1, 1] 范围，并转换为 int16 类型
signal = np.int16(signal / np.max(np.abs(signal)) * 32767)

# 播放音频
sd.play(signal, sample_rate)

# 等待音频播放完成
sd.wait()
