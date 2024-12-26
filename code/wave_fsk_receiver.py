import tkinter as tk
import pyaudio
import wave
import threading
from demodulate import demodulate_audio  # 假设 demodulate_signal_wav 是你用来解码 WAV 文件的函数

# 音频录制类
class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=48000,
                                  input=True,
                                  frames_per_buffer=1024)
        print("开始录制...")
        while self.is_recording:
            data = self.stream.read(1024)
            self.frames.append(data)

    def stop_recording(self, filename="output.wav"):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        # 保存音频文件
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))
        print(f"录制完毕，文件保存为 {filename}")


# 创建GUI
class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("音频录制")
        self.root.geometry("400x300")

        self.recorder = AudioRecorder()

        # 创建按钮
        self.start_button = tk.Button(root, text="开始录音", command=self.start_recording)
        self.start_button.pack(pady=20)

        self.stop_button = tk.Button(root, text="停止录音", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=20)

        # 创建文本框显示解码结果
        self.result_text = tk.Label(root, text="解码结果将显示在此", wraplength=300)
        self.result_text.pack(pady=20)

    def start_recording(self):
        # 禁用开始录音按钮，启用停止录音按钮
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # 在新线程中开始录音，避免阻塞主线程
        self.recording_thread = threading.Thread(target=self.recorder.start_recording)
        self.recording_thread.start()

    def stop_recording(self):
        # 禁用停止录音按钮，启用开始录音按钮
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # 停止录音并保存文件
        self.recorder.stop_recording("output/record.wav")
        self.recording_thread.join()  # 等待录音线程结束

        self.decode_and_display()

    def decode_and_display(self):
        # 假设 demodulate_signal_wav 函数返回解码后的文本
        decoded_text = demodulate_audio("output/record.wav")
        self.result_text.config(text=decoded_text)

        # 自动保存解码文本到文件
        self.save_decoded_text(decoded_text)

    def save_decoded_text(self, decoded_text):
        # 保存到文件
        with open("output/decoded_output.txt", "w") as f:
            f.write(decoded_text)

        print("解码文本已自动保存为 decoded_output.txt")


# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
