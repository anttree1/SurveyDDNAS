import matplotlib.pyplot as plt

# 指定横坐标和高度
categories = ['a', 'b', 'c', 'd', 'e']
heights = [15, 12, 14, 16, 13]
# 假设的方差列表
variances = [1, 0.5, 0.8, 1.2, 0.7]

# 设置条形的x坐标以分组，第一组有abc，第二组有de，带有间隙
x_coords = [0, 1, 2, 4, 5]

# 绘制条形图并添加误差线表示方差
plt.bar(x_coords, heights, color='skyblue', yerr=variances, capsize=5, tick_label=categories)

# 在组合下方添加文本
group_names = ['Group 1', 'Group 2']
# 计算每组中心坐标，用于文本标签的定位
group_x_coords = [1, 4.5]
for group_x, group_name in zip(group_x_coords, group_names):
    plt.text(group_x, -3, group_name, ha='center', va='bottom')  # 调整了y坐标以避免与x轴标签重叠

# 调整图表以便完整显示底部文本
plt.tight_layout()

# 显示图表
plt.show()