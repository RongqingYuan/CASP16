import matplotlib.pyplot as plt

# 你的字典数据
result = {'TS145': 35, 'TS393': 20, 'TS264': 18, 'TS462': 17, 'TS345': 16,
          'TS014': 12, 'TS051': 12, 'TS059': 8, 'TS015': 5, 'TS139': 4}
# 提取键和值
labels = list(result.keys())
values = list(result.values())

# 绘制条形图
plt.figure(figsize=(6, 4))
plt.bar(labels, values, color='skyblue')
# 添加标题和标签
plt.xlabel('groups')
plt.ylabel('# of targets')
plt.title('# of targets better than baseline_first (total 35)')

# 显示图表
plt.xticks(rotation=45)  # 旋转 x 轴标签，便于显示
plt.tight_layout()  # 调整布局
plt.show()
plt.savefig('./png/bar_plot_first.png', dpi=300)
