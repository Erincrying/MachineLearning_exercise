# pipeline

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 生成随机数据
# random.uniform(x, y)方法将随机生成一个实数，它在 [x,y] 范围内。
x = np.random.uniform(-3, 3, size = 100)
X = x.reshape(-1, 1) # shape：（100,1）

# np.random.normal()正态分布的噪音
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size = 100)

# 使用pipeline创建管道，送给poly_reg对象的数据会沿着管道的三步依次进行
poly_reg = Pipeline([ # Pipeline传入的是列表，列表中传入管道中每一步对应的类(这个类以元组的形式进行传送)
  ("poly", PolynomialFeatures(degree=2)), # 第一步：求多项式特征，相当于poly = PolynomialFeatures(degree=2)
  ("std_scaler", StandardScaler()), # 第二步：数值的均一化
  ("lin_reg", LinearRegression()) # 第三步：进行线性回归操作
])

# 将X送给poly_reg执行前两步操作，得到的全新的数据x会送给LinearRegression进行fit相应的操作
poly_reg.fit(X, y)
# predict操作也同上
y_predict = poly_reg.predict(X)

# 绘制
plt.scatter(x, y)
# 直接绘制，会导致图像错乱，因为x，y的值没有按照大小顺序排序
# plt.plot(x, y_predict2, color='r')
# argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
plt.show()