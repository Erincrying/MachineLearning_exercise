'''
  SVM中使用多项式特征
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures, StandardScaler # 引入多项式类、标准化
# 使用支持向量机的方法进行分类（线性SVM）
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline # 引入Pipeline顺序执行相关过程

X, y = datasets.make_moons(noise=0.15, random_state=666)
print(X.shape, 'X.shape')
print(y.shape, 'y.shape')

# 绘制样本
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
plt.show()

'''
  svm中使用多项式特征
  degree: 阶数
'''
def PolynomialSVC(degree, C = 1.0):
  # 使用pipeline创建管道，送给实例化对象的数据会沿着管道的三步依次进行
  return Pipeline([ # Pipeline传入的是列表，列表中传入管道中每一步对应的类(这个类以元组的形式进行传送)
    ("poly", PolynomialFeatures(degree=degree)), # 第一步：求多项式特征，相当于poly = PolynomialFeatures(degree=2)
    ("std_scaler", StandardScaler()), # 第二步：数值的均一化
    ("linearSVC", LinearSVC(C = C)) # 第三步：进行线性回归操作
  ])

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


poly_svc = PolynomialSVC(degree=3)
poly_svc.fit(X, y) # 这里也是不区分训练集和测试集，所以所有数据都用来训练
# 绘制决策边界
plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
# 绘制样本
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
plt.show()