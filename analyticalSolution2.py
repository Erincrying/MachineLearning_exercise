# 实现一维线性回归：解析解(求w，b)

import numpy as np
import matplotlib.pyplot as plt

# # 这里只是简单定义了5个值
# x = np.array([1., 2., 3., 4., 5.])
# y = np.array([1., 3., 2., 3., 5.])
# 增加数据量
# x轴数据
# x = np.arange(20)
x = np.random.uniform(0.0, 20.0, size=50)
# y轴数据
# y = np.array([0.4, 0.8, 1.1, 2.1, 2.8, 2.7, 3.5, 4.6, 5.1, 4.5, 6.0, 5.5, 6.9, 6.8, 7.6, 8.0, 8.8, 8.5, 9.5, 9.3])
y = x * 12 + np.random.normal(loc=0, scale=12.0, size=50)
print(y)
plt.scatter(x, y)
plt.show()

# x,y的均值
x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = 0.0 # w的分子
denominator_1 = 0.0 # w的分母的第一项
denominator_2 = 0.0 # w的分母的第二项(不包含1/m)
denominator = 0.0 # 分母

b_sum = 0.0
num = 0

for x_i, y_i in zip(x, y):
  numerator += y_i * (x_i - x_mean)
  denominator_1 += x_i ** 2
  denominator_2 += x_i
  num += 1

print(num)
# 求解w
denominator = denominator_1 - 1 / num * (denominator_2 ** 2)
w = numerator / denominator
# 求解b
for x_i, y_i in zip(x, y):
  b_sum += y_i - w * x_i
b = 1 / num * b_sum

y_hat = w * x + b # 通过刚刚求解的w,b，基于每一个x所属的特征值进行预测

plt.scatter(x, y) # 通过散点图的方式把x，y绘制出来
plt.plot(x, y_hat, color = 'r') # 绘制直线(x采用刚刚同样的x值,y采用刚刚预测的y_hat)
# plt.axis([0, 25, 0, 210]) # 规定坐标轴范围
plt.show()