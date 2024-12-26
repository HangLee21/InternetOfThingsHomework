import os
import tkinter as tk


from APP import AudioRecorderApp


# use_file_input = False  # 设置为 False 来实时接收音频信号
root = tk.Tk()
app = AudioRecorderApp(root)
root.mainloop()




