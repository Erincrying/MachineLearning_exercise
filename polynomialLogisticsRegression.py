# 逻辑回归问题：在逻辑回归中使用多项式特征

import numpy as np
import matplotlib.pyplot as plt
from classLogisticsRegression import logisticsRegression # 引入逻辑回归类对象
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

'''
  绘制决策边界
  params-model:训练好的model
  params-axis:绘制区域坐标轴范围（0,1,2,3对应x轴和y轴的范围）
'''
def plot_decision_boundary(model, axis):
  # meshgrid:生成网格点坐标矩阵
  x0, x1 = np.meshgrid(
    # 通过linspace把x轴分成无数点
    # axis[1] - axis[0]是x的左边界减去x的右边界
    # axis[3] - axis[2]：y的最大值减去y的最小值
        
    # arr.shape    # (a,b)
    # arr.reshape(m,-1) #改变维度为m行、d列 （-1表示列数自动计算，d= a*b /m）
    # arr.reshape(-1,m) #改变维度为d行、m列 （-1表示行数自动计算，d= a*b /m ）
    np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
    np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
  )
  print('x0', x0)
  # print('x1', x1)
  # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，相加后列数不变。
  # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，相加后行数不变。
  # .ravel():将多维数组转换为一维数组
  X_new = np.c_[x0.ravel(), x1.ravel()]
  y_predict = model.predict(X_new)
  
  # 这里不能zz = y_predict.reshape(x0.shape)，会报错'list' object has no attribute 'reshape'
  # 要通过np.array转换一下
  zz = np.array(y_predict).reshape(x0.shape)

  from matplotlib.colors import ListedColormap
  # ListedColormap允许用户使用十六进制颜色码来定义自己所需的颜色库，并作为plt.scatter()中的cmap参数出现：
  custom_cmap = ListedColormap(['#F5FFFA', '#FFF59D', '#90CAF9'])
  # coutourf([X, Y,] Z,[levels], **kwargs),contourf画的是登高线之间的区域
  # Z是和X,Y相同维数的数组。
  plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)




np.random.seed(666) # 添加随机种子
X = np.random.normal(0, 1, size=(200, 2)) # 均值为0，标准差为1的随机序列。200个样本，每个样本2个特征
# 布尔类型转换成int类型，true对应1，false对应0
y = np.array(X[:, 0]**2 + X[:, 1]**2 < 1.5, dtype='int') # X的第一个特征（第一列）的平方+X的第二个特征（第二列）的平方，半径值1.5

# 绘制样本
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
plt.show()

# # 使用逻辑回归多多项式项进行分类划分
# # 实例化
# log_reg = logisticsRegression()
# # 这里就不划分训练样本和测试样本了，这里直接用所有样本进行训练
# log_reg.fit(X, y)
# # 绘制决策边界
# plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4]) # x、y轴的范围大致都在（-4,4）
# # 绘制样本
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
# plt.show()


# 包装管道进行多项式回归（传入degree，返回多项式回归的类）
def PolynomialLogisticsRegression(degree):
  # 使用pipeline创建管道，送给poly_reg对象的数据会沿着管道的三步依次进行
  return Pipeline([ # Pipeline传入的是列表，列表中传入管道中每一步对应的类(这个类以元组的形式进行传送)
    ("poly", PolynomialFeatures(degree=degree)), # 第一步：求多项式特征，相当于poly = PolynomialFeatures(degree=2)
    ("std_scaler", StandardScaler()), # 第二步：数值的均一化
    ("log_reg", logisticsRegression()) # 第三步：进行逻辑回归操作
  ])


''' degree=2'''
poly_log_reg = PolynomialLogisticsRegression(2)
poly_log_reg.fit(X, y)
# 查询分类准确度
print('degree=2分类准确度', poly_log_reg.score(X, y))
# 绘制决策边界
plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4]) # x、y轴的范围大致都在（-4,4）
# 绘制样本
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
plt.show()


''' degree=20 '''
poly_log_reg2 = PolynomialLogisticsRegression(20)
poly_log_reg2.fit(X, y)
# 查询分类准确度
print('degree=20分类准确度', poly_log_reg2.score(X, y))
# 绘制决策边界
plot_decision_boundary(poly_log_reg2, axis=[-4, 4, -4, 4]) # x、y轴的范围大致都在（-4,4）
# 绘制样本
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
plt.show()