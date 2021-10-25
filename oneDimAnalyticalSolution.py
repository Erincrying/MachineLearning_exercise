# 实现一维线性回归：解析解（求a、b）

import numpy as np
import matplotlib.pyplot as plt

# 这里只是简单定义了5个值
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

plt.scatter(x, y)
plt.show()

# x,y的均值
x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = 0.0 # 分子
denominator = 0.0 # 分母

for x_i, y_i in zip(x, y):
  numerator += (x_i - x_mean) * (y_i - y_mean)
  denominator += (x_i - x_mean) ** 2

# 求解a
a = numerator / denominator
# 求解b
b = y_mean - a * x_mean

y_hat = a * x + b # 通过刚刚求解的a,b，基于每一个x所属的特征值进行预测

plt.scatter(x, y) # 通过散点图的方式把x，y绘制出来
plt.plot(x, y_hat, color = 'r') # 绘制直线(x采用刚刚同样的x值,y采用刚刚预测的y_hat)
plt.axis([0, 6, 0, 6]) # 规定坐标轴范围
plt.show()