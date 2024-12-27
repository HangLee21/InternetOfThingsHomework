import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_ber_relationship():
    # 数据
    distances = np.array([50, 100, 150, 200])  # 距离
    ber_dist = np.array([0.1, 0.2, 0.3, 0.4])  # 对应的BER

    obstruction_factors = np.array([0, 2, 4])  # 遮挡因子
    obstruction_labels = ['无', '书籍', '人体']  # 对应中文标签
    ber_obs = np.array([0.05, 0.1, 0.15])  # 对应的BER

    noise_levels = np.array([0, 1, 2])  # 噪声水平
    noise_labels = ['静环境', '人声说话环境', '大音量音乐嘈杂环境']  # 对应中文标签
    ber_noise = np.array([0.05, 0.1, 0.2])  # 对应的BER

    # 创建一个3个子图的图形
    plt.figure(figsize=(15, 5))

    # 距离与BER的关系
    plt.subplot(1, 3, 1)
    plt.plot(distances, ber_dist, marker='o', color='b', label='距离与BER关系')
    plt.xlabel('距离 (m)')
    plt.ylabel('误码率 (BER)')
    plt.title('距离与BER的关系')
    plt.grid(True)
    plt.legend()

    # 遮挡与BER的关系
    plt.subplot(1, 3, 2)
    plt.plot(obstruction_factors, ber_obs, marker='o', color='r', label='遮挡与BER关系')
    plt.xticks(obstruction_factors, obstruction_labels)  # 设置x轴为中文标签
    plt.xlabel('遮挡因子')
    plt.ylabel('误码率 (BER)')
    plt.title('遮挡与BER的关系')
    plt.grid(True)
    plt.legend()

    # 噪声与BER的关系
    plt.subplot(1, 3, 3)
    plt.plot(noise_levels, ber_noise, marker='o', color='g', label='噪声与BER关系')
    plt.xticks(noise_levels, noise_labels)  # 设置x轴为中文标签
    plt.xlabel('噪声环境')
    plt.ylabel('误码率 (BER)')
    plt.title('噪声与BER的关系')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()  # 自动调整子图之间的距离
    plt.show()

# 调用绘图函数
plot_ber_relationship()
