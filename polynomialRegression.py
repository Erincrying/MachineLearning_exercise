# 多项式回归

import numpy as np
import matplotlib.pyplot as plt
from classLinearRegression import LinearRegression # 引入线性回归类对象

# 生成随机数据
# random.uniform(x, y)方法将随机生成一个实数，它在 [x,y] 范围内。
x = np.random.uniform(-3, 3, size = 100)
X = x.reshape(-1, 1) # shape：（100,1）

# np.random.normal()正态分布的噪音
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size = 100)

# 绘制样本
plt.scatter(x, y)
plt.show()



'''
  线性回归拟合数据集
  不划分测试集与训练集，所有样本送去训练
'''
# 实例化对象
lin_reg = LinearRegression()
# 训练
lin_reg.fit_normal(X, y)
y_predict = lin_reg.predict(X)
# 预测
print(y_predict, 'y_predict')
# 绘制预测后的结果
plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()


'''
  多项式回归
'''
# np.vstack():在竖直方向上堆叠
# np.hstack():在水平方向上平铺
X2 = np.hstack([X, X**2])
# 实例化新对象
lin_reg2 = LinearRegression()
lin_reg.fit_normal(X2, y)
y_predict2 = lin_reg.predict(X2)
print(y_predict2, 'y_predict2多项式回归预测结果')
# 绘制多项式回归，预测后的结果
plt.scatter(x, y)
# 直接绘制，会导致图像错乱，因为x，y的值没有按照大小顺序排序
# plt.plot(x, y_predict2, color='r')
# argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()
