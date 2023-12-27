import matplotlib.pyplot as plt
import numpy as np

data = [[67.4, 64.2], [67.6, 65.7], [68.4, 65.3], [68.9, 66.3], [60, 60], [70.4, 67.2], [71.2, 68.7], [68.9, 66.7],
        [69.2, 67.1], [72.3, 69.5], [71.5, 68.4], [71.9, 70.8], [70.9, 68.0]]

# 设置每个部分的宽度和间隙
bar_width = 0.35
spacing = 0.05

# 创建画布和子图
fig, ax = plt.subplots()

# 设置x轴的刻度和标签
index = np.arange(len(data))
xticks = index + (bar_width + spacing) / 2
ax.set_xticks(xticks)
ax.set_xticklabels(
    ['[117]', '[105]', '[37]', '[40]', '', '[51]', '[130]', '[61]', '[118]', '[135]', '[59]', '[144]', '[34]'],
    fontsize=8)
ax.set_ylim(60, 75)
ax.set_ylabel('Accuracy/%')

# 绘制两个长方形
for i in range(len(data)):
    if i == 0:
        ax.bar(index[i] - bar_width / 2, data[i][0], bar_width, label='Close eyes', color='#EF3E42')
        ax.bar(index[i] + bar_width / 2, data[i][1], bar_width, label='Open eyes', color='#005A9C')
    else:
        ax.bar(index[i] - bar_width / 2, data[i][0], bar_width, color='#EF3E42')
        ax.bar(index[i] + bar_width / 2, data[i][1], bar_width, color='#005A9C')
# 添加图例
ax.legend()
ax.legend()
plt.savefig("blink_noise.pdf", format='pdf')
plt.show()
