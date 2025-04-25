import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
labels = ['学业压力', '个人兴趣', '劳动报酬', '家庭经济情况', '服务校园师生', '心情', '班次人数限制', '年级']
values = [12, 7, 5, 5, 5, 1, 1, 5]
explode = [0.05] * len(values)  # 每个扇区稍微“拉出”一点

# 绘图
plt.figure(figsize=(9, 9))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode, textprops={'fontsize': 12})
plt.title('影响勤工工作时长的因素', fontsize=16)
plt.axis('equal')
plt.tight_layout()
plt.show()