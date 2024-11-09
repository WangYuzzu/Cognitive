import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 使用 Seaborn 的默认主题，并指定字体
sns.set_theme(style='whitegrid', font='SimHei', font_scale=1.0)

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 准备数据
labels = ['苹果', '香蕉', '橘子', '葡萄', '其他']
sizes = [25, 20, 15, 10, 30]

# 创建一个绘图区域
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制饼图
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontproperties': 'SimHei'}
)

# 设置标题
ax.set_title('水果销售比例', fontproperties='SimHei')

# 确保图形为圆形
ax.axis('equal')

# 显示图形
plt.show()
