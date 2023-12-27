import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 生成示例EEG数据
# 注意：这里的数据只是随机生成的示例数据，并不代表真实的EEG数据

data = loadmat(r'F:\data\sch1EEG\C\0.mat')
# 获取数据和标签

a = data['data']
num_channels = 16  # 通道数
num_samples = len(a[1])  # 样本数
eeg_data = np.array(a)

plt.figure(figsize=(10, 6))
plt.title("EEG Data")
plt.xlabel("Sample")
plt.ylabel("Channel")
channel_name = ['','F7','F3','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']

# 计算y轴的范围
y_min = np.min(eeg_data)
y_max = np.max(eeg_data)
y_range = y_max - y_min

# 定义平移和缩放系数
translate_factor = 3  # 平移系数
scale_factor = 5  # 缩放系数

# 绘制每个通道的曲线，增加Y轴上的平移幅度
for channel in range(num_channels):
    # 将每个通道的数据在y轴上平移和缩放，使得能够更清晰地显示各个通道的波形
    eeg_channel = (eeg_data[channel] - y_min) / y_range * scale_factor + translate_factor * channel

    # 绘制每个通道的曲线，使用相同颜色和线条粗细为2
    plt.plot(eeg_channel, linewidth=0.2)

# 设置y轴的刻度表示每个通道
plt.yticks(np.arange(17) * translate_factor, channel_name)
# 展示EEG数据图像
plt.savefig("sine_wave.pdf", format='pdf')
plt.show()
