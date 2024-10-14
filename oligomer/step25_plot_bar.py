import matplotlib.pyplot as plt
greater = {'TS393': 17, 'TS264': 14, 'TS462': 12, 'TS345': 11,
           'TS051': 9, 'TS014': 8, 'TS059': 5, 'TS015': 3, 'TS139': 2}

less = {'TS139': 29, 'TS015': 28, 'TS059': 26, 'TS014': 21, 'TS051': 19,
        'TS345': 16, 'TS264': 15, 'TS393': 13, 'TS462': 11}

# 定义你的字典
greater = {'TS393': 17, 'TS264': 14, 'TS462': 12, 'TS345': 11,
           'TS051': 9, 'TS014': 8, 'TS059': 5, 'TS015': 3, 'TS139': 2}

less = {'TS139': 29, 'TS015': 28, 'TS059': 26, 'TS014': 21, 'TS051': 19,
        'TS345': 16, 'TS264': 15, 'TS393': 13, 'TS462': 11}
zero = {'TS015': 15, 'TS139': 11, 'TS059': 8, 'TS014': 6, 'TS345': 6,
        'TS051': 6, 'TS393': 2, 'TS264': 1, 'TS462': 1}

# 计算每个 key 的比例 greater / (greater + less)
ratios = {key: greater[key] / (greater[key] + less[key]-zero[key])
          for key in greater}
# sort by value
ratios = dict(sorted(ratios.items(), key=lambda item: item[1], reverse=True))
# 提取键和值
labels = list(ratios.keys())
values = list(ratios.values())

# 绘制条形图
plt.figure(figsize=(6, 4))
plt.bar(labels, values, color='skyblue')

# 添加标题和标签
plt.xlabel('TS Labels')
plt.ylabel('Ratio')
plt.title(' > 1.01 / (> 1.01 + <0.99)')

# 显示图表
plt.xticks(rotation=45)  # 旋转 x 轴标签，便于显示
plt.tight_layout()  # 调整布局
plt.savefig('./png/bar_plot_ratio.png', dpi=300)
