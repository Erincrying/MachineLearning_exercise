# scikit-learn中的多项式回归（一个特征值）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 生成随机数据
# random.uniform(x, y)方法将随机生成一个实数，它在 [x,y] 范围内。
x = np.random.uniform(-3, 3, size = 100)
X = x.reshape(-1, 1) # shape：（100,1）

# np.random.normal()正态分布的噪音
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size = 100)

# 实例化对象
# degree表示要为原本的数据集添加最多几次幂相应的特征
poly = PolynomialFeatures(degree=2)
# 训练
poly.fit(X)
# 特征值的转换
X2 = poly.transform(X)



'''
  线性回归拟合数据集
  这里调用sklearn的LinearRegression类
'''
# 实例化新对象
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)
# 绘制多项式回归，预测后的结果
plt.scatter(x, y)
# 直接绘制，会导致图像错乱，因为x，y的值没有按照大小顺序排序
# plt.plot(x, y_predict2, color='r')
# argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()

print(lin_reg2.coef_, 'lin_reg2.coef系数')
print(lin_reg2.intercept_, 'lin_reg2.intercept_截距')
