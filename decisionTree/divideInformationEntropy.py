# 使用信息熵寻找最优划分

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from math import log

# 这里先用鸢尾花数据集（150行4列：150个样本，4个特征值）
iris = datasets.load_iris()

X = iris.data[:, 2:] # 只取两个维度的数据特征（方便可视化，保留后两个特征）
y = iris.target

'''
  创建决策树
  max_depth: 决策树最高深度
  entropy: 熵
'''
# 训练决策树分类器
dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
dt_clf.fit(X, y)

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

# 绘制决策边界
plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3]) # x、y轴的范围
# 样本
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.show()

'''
  模拟使用信息熵进行划分
  d:划分维度
  value：阈值
''' 
def split(X, y ,d, value):
  index_a = (X[:, d] <= value) # 定义索引
  index_b = (X[:, d] >value)
  return X[index_a], X[index_b], y[index_a], y[index_b]


'''
  求信息熵
'''
def entropy(y):
  # 将y值做成字典
  # counter包含键值对，y的取值-y的取值对应的分类个数
  counter = Counter(y)
  res = 0.0
  # 遍历看每一个不同的类别，有多少个样本点
  for num in counter.values():
    p = num / len(y)
    res += -p * log(p)
  return res

'''
  划分使信息熵最低
  在d列对应特征值的数据寻找信息熵和最小的划分方式
'''
def try_spilt(X, y):
  best_entropy = float('inf')
  best_d, best_v = -1, -1 # 维度、阈值
  # 穷搜
  for d in range(X.shape[1]): # 有多少列（特征），shape[0]有多少行，shape[1]有多少列
    sorted_index = np.argsort(X[:, d]) # 返回第d列数据排序后相应的索引
    for i in range(1, len(X)): # 对每一个样本进行遍历
      if (X[sorted_index[i-1], d] != X[sorted_index[i], d]):
        # d这个维度上从1开始，找i-1 和i的中间值
        # 可选值是在d这个维度上的中间值
        v = (X[sorted_index[i-1], d] + X[sorted_index[i], d]) / 2
        # X_l左子树，X_r右子树
        # 进行划分
        X_l, X_r, y_l, y_r = split(X, y ,d, v)
        #  信息熵的和
        e = entropy(y_l) + entropy(y_r)
        # best_entropy:之前搜索过的某一个信息熵
        if e < best_entropy: # 找到更好的划分方式
          best_entropy, best_d, best_v = e, d, v
  return best_entropy, best_d, best_v
      
'''
  调用
'''
best_entropy, best_d, best_v = try_spilt(X, y)
print('best_entropy=', best_entropy)
print('best_d=', best_d)
print('best_v=', best_v)

# 第一次划分得到的数据
X1_l, X1_r, y1_l, y1_r = split(X, y ,best_d, best_v)
print(entropy(y1_l), 'y1_l的信息熵')
print(entropy(y1_r), 'y1_r的信息熵')

# y1_l对应划分之后左边的部分，信息熵为0，不需要再进行划分
# y1_r对应划分之后右边的部分，信息熵大于0，可以继续进行划分
best_entropy2, best_d2, best_v2 = try_spilt(X1_r, y1_r)
print('best_entropy2=', best_entropy2)
print('best_d2=', best_d2)
print('best_v2=', best_v2)
# 第二次划分得到的数据
X2_l, X2_r, y2_l, y2_r = split(X1_r, y1_r ,best_d2, best_v2)
print(entropy(y2_l), 'y2_l的信息熵')
print(entropy(y2_r), 'y2_r的信息熵')