# 过拟合与欠拟合
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 引入线性回归类对象
from sklearn.metrics import mean_squared_error # 引入均方误差

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

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
  线性回归拟合二次函数（欠拟合）
'''
# 实例化对象
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict = lin_reg.predict(X)
print('线性回归预测与真值的均方误差', mean_squared_error(y, y_predict))
# 绘制预测后的结果
plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()

'''
  多项式回归拟合二次函数
'''
# 包装管道进行多项式回归（传入degree，返回多项式回归的类）
def polynomialRegression(degree):
  # 使用pipeline创建管道，送给poly_reg对象的数据会沿着管道的三步依次进行
  return Pipeline([ # Pipeline传入的是列表，列表中传入管道中每一步对应的类(这个类以元组的形式进行传送)
    ("poly", PolynomialFeatures(degree=degree)), # 第一步：求多项式特征，相当于poly = PolynomialFeatures(degree=2)
    ("std_scaler", StandardScaler()), # 第二步：数值的均一化
    ("lin_reg", LinearRegression()) # 第三步：进行线性回归操作
  ])

''' degree= 2 '''
poly2_reg = polynomialRegression(2)
# 将X送给poly_reg执行前两步操作，得到的全新的数据x会送给LinearRegression进行fit相应的操作
poly2_reg.fit(X, y)
y2_predict = poly2_reg.predict(X)
print('degree=2多项式回归预测与真值的均方误差', mean_squared_error(y, y2_predict))
# 绘制预测后的结果
plt.scatter(x, y)
plt.plot(np.sort(x), y2_predict[np.argsort(x)], color='r')
plt.show()

''' degree= 10 '''
poly10_reg = polynomialRegression(10)
# 将X送给poly_reg执行前两步操作，得到的全新的数据x会送给LinearRegression进行fit相应的操作
poly10_reg.fit(X, y)
y10_predict = poly10_reg.predict(X)
print('degree=10多项式回归预测与真值的均方误差', mean_squared_error(y, y10_predict))
# 绘制预测后的结果
plt.scatter(x, y)
plt.plot(np.sort(x), y10_predict[np.argsort(x)], color='r')
plt.show()

''' degree= 100 '''
poly100_reg = polynomialRegression(100)
# 将X送给poly_reg执行前两步操作，得到的全新的数据x会送给LinearRegression进行fit相应的操作
poly100_reg.fit(X, y)
y100_predict = poly100_reg.predict(X)
print('degree=100多项式回归预测与真值的均方误差', mean_squared_error(y, y100_predict))
# 绘制预测后的结果
plt.scatter(x, y)
plt.plot(np.sort(x), y100_predict[np.argsort(x)], color='r')
plt.show()
# 准确画出图像
X_plot = np.linspace(-3, 3, 100).reshape(100, 1) # 通过linspace在（-3,3）直接均匀取值100个，然后通过reshape变成一个二维矩阵
y_plot = poly100_reg.predict(X_plot)
# 现在绘制的图形会更加准确，因为x的取值是在（-3,3）均匀取值的，所以不会出现两个点之间相隔太大的情况
plt.scatter(x, y)
plt.plot(X_plot[:, 0], y_plot, color='r')
plt.axis([-3, 3, -1, 10])
plt.show()