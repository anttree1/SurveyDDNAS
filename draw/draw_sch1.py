import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 指定横坐标和高度
categories = ['[94]', '[92]', '[56]', '[60]', '[93]', '[69]', '[121]', '[138]']
heights = [69.91, 67.62, 63.70, 65.24, 65.44, 68.23, 66.25, 63.19]
hot_size = [7, 12, 12, 8, 5, 6, 8, 9]
# 假设的方差列表
variances = [2.03, 1.19, 2.09, 1.04, 1.21, 1.77, 3.64, 1.52]

# 使用热力图的颜色
cmap = plt.cm.hot
norm = mcolors.Normalize(vmin=1, vmax=15)
colors = [cmap(norm(value)) for value in hot_size]

# 设置y轴的名称
plt.ylabel('Accuracy / %')

# 控制y轴的范围，例如从0到20
plt.ylim(60, 75)

# 设置条形的x坐标以分组，第一组有abc，第二组有de，带有间隙
x_coords = [0, 1, 3, 4, 5, 7, 8, 9]

# 绘制条形图并添加误差线表示方差
plt.bar(x_coords, heights, color=colors, yerr=variances, capsize=5, tick_label=categories)

# 在组合下方添加文本
group_names = ['EA', 'RA', 'DARTS']
# 计算每组中心坐标，用于文本标签的定位
group_x_coords = [0.5, 4, 8]
for group_x, group_name in zip(group_x_coords, group_names):
    plt.text(group_x, -3, group_name, ha='center', va='bottom')  # 调整了y坐标以避免与x轴标签重叠

# 调整图表以便完整显示底部文本
plt.tight_layout()

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm)

# 显示图表
plt.show()
