# scikit-learn中的多项式回归（多个特征值）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 生成数据集
X = np.arange(1, 11).reshape(-1, 2) # arange函数用于创建等差数组，从1-10取值，设置成5行两列的矩阵



# # 实例化对象(degree=2)
# # degree表示要为原本的数据集添加最多几次幂相应的特征
# poly = PolynomialFeatures(degree=2)
# # 训练
# poly.fit(X)
# # 特征值的转换
# X2 = poly.transform(X)
# print(X2)




# 实例化对象(degree=3)
# degree表示要为原本的数据集添加最多几次幂相应的特征
poly2 = PolynomialFeatures(degree=3)
# 训练
poly2.fit(X)
# 特征值的转换
X3 = poly2.transform(X)
print(X3.shape, 'X3.shape')