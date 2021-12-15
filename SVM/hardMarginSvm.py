'''
  Hard Margin SVM
'''

from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# 使用支持向量机的方法进行分类（线性SVM）
from sklearn.svm import LinearSVC

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


'''
  绘制svc决策边界根据margin计算的两根线
  params-model:训练好的model
  params-axis:绘制区域坐标轴范围（0,1,2,3对应x轴和y轴的范围）
'''
def plot_svc_decision_boundary(model, axis):
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
  # print('x0', x0)
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

  w = model.coef_[0] # coef_是二维数组
  b = model.intercept_[0]
  # 此时的决策边界应该是w0*x0+w1*x1+b=0
  # 上下的两根直线方程分别为w0*x0+w1*x1+b=1， w0*x0+w1*x1+b=-1
  # 为了方便可视化，以x1为纵轴，x0为横轴改写上面的直线方程
  # 决策边界：x1 = -w0/w1 * x0 - b/w1
  # 上下两根线：x1 = -w0/w1 * x0 - b/w1 + 1/w1；x1 = -w0/w1 * x0 - b/w1 - 1/w1
  plot_x = np.linspace(axis[0], axis[1], 200)
  # 接下来求对应x相应对上下两根线的y值
  up_y = -w[0]/w[1] * plot_x - b/w[1] + 1/w[1]
  down_y = -w[0]/w[1] * plot_x - b/w[1] - 1/w[1]
  # up_y、down_y有可能超过了axis规定的y轴的范围，需要对数据进行过滤
  up_index = (up_y >= axis[2]) & (up_y <= axis[3]) # 对应布尔数组
  down_index = (down_y >= axis[2]) & (down_y <= axis[3])
  # 绘制
  plt.plot(plot_x[up_index], up_y[up_index], color='red')
  plt.plot(plot_x[down_index], down_y[down_index], color='red')



'''
  生成样本
'''

# 这里先用鸢尾花数据集（150行4列：150个样本，4个特征值）
iris = datasets.load_iris()

X = iris.data
y = iris.target

# 这里鸢尾花数据集有三种分类，我们先把数据集做成只有两种分类（二分类）
X = X[y < 2, :2] # 取前两个特征方便可视化
y = y[y < 2]


# 绘制y=0、y=1相应的x的两个特征在二维平面的坐标,[y == 行范围, 列范围]
# X[y == 0, 1]：获取y==0的行，然后获取这些行的第二个元素
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
plt.show()


'''
  数据标准化
'''
standardScaler = StandardScaler()
# 这里只是为了数据可直观化（方便看清svm的分类以及软间隔C取值不同的结果），省去了训练集训练与测试的过程
standardScaler.fit(X)
X_standardScaler = standardScaler.transform(X)

'''
  调用SVM（线性SVM）
  C:超参数（取值越大越偏向硬间隔，取值越小容错空间越大）
'''
svc = LinearSVC(C=1e9)
svc.fit(X_standardScaler, y)

# 绘制决策边界
plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
# 绘制样本
plt.scatter(X_standardScaler[y == 0, 0], X_standardScaler[y == 0, 1], color = "orange")
plt.scatter(X_standardScaler[y == 1, 0], X_standardScaler[y == 1, 1], color = "pink")
plt.show()

'''
  重新实例化，减小超参数C
'''
svc2 = LinearSVC(C=0.01)
svc2.fit(X_standardScaler, y)

# 绘制决策边界
plot_decision_boundary(svc2, axis=[-3, 3, -3, 3])
# 绘制样本
plt.scatter(X_standardScaler[y == 0, 0], X_standardScaler[y == 0, 1], color = "orange")
plt.scatter(X_standardScaler[y == 1, 0], X_standardScaler[y == 1, 1], color = "pink")
plt.show()


print(svc.coef_, 'svc系数值') # svc的系数值
print(svc.intercept_, 'svc截距') # svc的系数值


# svc绘制决策边界以及两根直线
plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
# 绘制样本
plt.scatter(X_standardScaler[y == 0, 0], X_standardScaler[y == 0, 1], color = "orange")
plt.scatter(X_standardScaler[y == 1, 0], X_standardScaler[y == 1, 1], color = "pink")
plt.show()

# svc2绘制决策边界以及两根直线
plot_svc_decision_boundary(svc2, axis=[-3, 3, -3, 3])
# 绘制样本
plt.scatter(X_standardScaler[y == 0, 0], X_standardScaler[y == 0, 1], color = "orange")
plt.scatter(X_standardScaler[y == 1, 0], X_standardScaler[y == 1, 1], color = "pink")
plt.show()