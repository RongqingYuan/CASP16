
import matplotlib.pyplot as plt

# 你的字典数据
result = {'TS264': 32, 'TS014': 29, 'TS462': 23, 'TS345': 22, 'TS051': 21, 'TS015': 17, 'TS139': 8, 'TS059': 6,
          # 'TS145': 0
          }

# 提取键和值
labels = list(result.keys())
values = list(result.values())

# 绘制条形图
plt.figure(figsize=(6, 4))
plt.bar(labels, values, color='skyblue')
# 添加标题和标签
plt.xlabel('groups')
plt.ylabel('# of targets')
plt.title(
    '# of targets better than baseline_first (total 59)')

# 显示图表
plt.xticks(rotation=45)  # 旋转 x 轴标签，便于显示
plt.tight_layout()  # 调整布局
plt.show()
plt.savefig('./png/bar_plot_first.png', dpi=300)
