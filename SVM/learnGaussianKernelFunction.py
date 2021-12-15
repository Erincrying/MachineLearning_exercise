'''
  直观理解高斯核函数
'''
import numpy as np
import matplotlib.pyplot as plt

# 每个样本只有一个特征，并且线性不可分
x = np.arange(-4, 5, 1) # 前4后5.步长为1，对于x而言只有一个特征

y = np.array((x >= -2) & (x <= 2), dtype='int')# 线性不可分的分类数据(x>=-2并且x<=2区间的点，值等于1，其他等于0)

# print(x, y)
# 绘制样本
plt.scatter(x[y == 0], [0]*len(x[y==0]))
plt.scatter(x[y == 1], [0]*len(x[y==1]))
plt.show()

'''
  将一维数据映射到二维
  l:landmark地标
'''
def gaussian(x, l):
  gamma = 1.0
  return np.exp(-gamma * (x - l) ** 2) # 这里x和l都是一维数据，他们的模直接用x-l

l1, l2 = -1, 1

# 存储新的二维数据
X_new = np.empty((len(x), 2))

for i, data in enumerate(x):
  X_new[i, 0] = gaussian(data, l1) # 第0个特征
  X_new[i, 1] = gaussian(data, l2) # 第1个特征

# 可视化结果
plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1])
plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1])
plt.show()
