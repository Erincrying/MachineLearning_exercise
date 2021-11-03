# 逻辑回归问题；在多元线性回归问题的基础上进行修改
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from manyDimLinearRegression import X_train

class logisticsRegression:
  def __init__(self):
      # 初始化logistics Regression模型
      self.coef_ = None # 系数，对应theta1-n，对应的向量
      self.interception_ = None # 截距，对应theta0
      self._theta = None # 定义私有变量，整体计算的theta

  # sigmoid函数数据溢出问题：https://blog.csdn.net/wofanzheng/article/details/103976889
  # 定义私有sigmoid函数
  # def _sigmoid(self, t):
  #   return 1. / 1. + np.exp(-t)

  def _sigmoid(self, x):
    l=len(x)
    y=[]
    for i in range(l):
      if x[i]>=0:
        y.append(1.0/(1+np.exp(-x[i])))
      else:
        y.append(np.exp(x[i])/(np.exp(x[i])+1))
    return y

  '''
    梯度下降
  '''
  def fit(self, X_train, y_train, eta = 5.0, n_iters = 1e4): 
    # 根据训练数据集X_train, y_ .train训练logistics Regression模型
    # X_train的样本数量和y_train的标记数量应该是一致的
    # 使用shape[0]读取矩阵第一维度的长度，在这里就是列数
    assert X_train.shape[0] == y_train.shape[0], \
    "the size of x_ .train must be equal to the size of y_ train"
    # 损失函数
    def J(theta, X_b, y):
      y_hat = self._sigmoid(X_b.dot(theta))
      try:
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
      except:
        return float('inf') # 返回float最大值

    # 梯度(比较笨的方法)
    def dJ(theta, X_b, y):
      return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)


    # 梯度下降求解theta矩阵
    def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon = 1e-8):
      theta = initial_theta
      cur_iters = 0

      while cur_iters < n_iters:
          gradient = dJ(theta, X_b, y) # 求梯度
          last_theta = theta # theta重新赋值前，记录上一场的值
          theta = theta - eta * gradient # 通过一定的eta学习率取得下一个点的theta
          # 最近两点的损失函数差值小于一定精度，退出循环
          if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
              break
          cur_iters += 1
      return theta

    # 得到X_b
    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    initial_theta = np.zeros(X_b.shape[1]) # 设置n+1维的向量,X_b.shape[1]:第一行的维数
    # X_b.T是X_b的转置,.dot是点乘,np.linalg.inv是求逆
    # 获取theta
    self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

    self.interception_ = self._theta[0] # 截距
    self.coef_ = self._theta[1:] # 系数 
    return self

  '''
    预测可能性的过程
  '''
  def predict_prob(self, X_predict):
    # 给定待预测数据集X_predict,返回表示X_predict的结果概率向量
    X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
    return self._sigmoid(X_b.dot(self._theta))


  '''
    预测过程
  '''
  def predict(self, X_predict):
    # 给定待预测数据集X_predict,返回表示X_predict的结果向量
    prob = self.predict_prob(X_predict) # prob向量存储的都是0-1的浮点数
    # 进行分类(布尔类型强制转换为整型)
    # return np.array(prob >= 0.5, dtype='int')
    l = len(prob)
    temp_prob=[]
    for i in range(l):
      if prob[i] >= 0.5:
        temp_prob.append(1)
      else:
        temp_prob.append(0)
    return temp_prob

  '''
    显示属性
  '''
  def __repr__(self):
      return "logisticsRegression()"



# 这里先用鸢尾花数据集（150行4列：150个样本，4个特征值）
iris = datasets.load_iris()

X = iris.data
y = iris.target

# 这里鸢尾花数据集有三种分类，我们先把数据集做成只有两种分类
X = X[y < 2, :2] # 取前两个特征方便可视化
y = y[y < 2]

# print('X', X.shape)
# print('y', y.shape)

# 绘制y=0、y=1相应的x的两个特征在二维平面的坐标,[y == 行范围, 列范围]
# X[y == 0, 1]：获取y==0的行，然后获取这些行的第二个元素
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
plt.show()

# 训练
# X_train = X[y < 2, :2] # 取前两个特征方便可视化
# y_train = y[y < 2]
# 实例化
log_reg = logisticsRegression()
# 分离测试集与数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# print('X_train', X_train)
# print('X_test', X_test)
# print('y_train', y_train)
# print('y_test', y_test)


# 进行训练,通过测试训练分离的方法测试逻辑回归的结果
log_reg.fit(X_train, y_train) # logisticsRegression()
# log_reg.predict_prob(X_test)
print('X_test概率值', log_reg.predict_prob(X_test))
print('通过概率划分得到的分类值', log_reg.predict(X_test))
print('y_test测试值', y_test)

# 绘制决策边界的直线(全数据集)
x1_plot = np.linspace(4, 8, 1000)
x2_plot = (-log_reg.coef_[0] * x1_plot - log_reg.interception_) / log_reg.coef_[1]
plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "pink")
plt.plot(x1_plot, x2_plot)
plt.show()

# 绘制决策边界的直线(测试数据集)
x1_plot = np.linspace(4, 8, 1000)
x2_plot = (-log_reg.coef_[0] * x1_plot - log_reg.interception_) / log_reg.coef_[1]
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color = "orange")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color = "pink")
plt.plot(x1_plot, x2_plot)
plt.show()

# 绘制决策边界的直线(训练数据集)
# x1_plot = np.linspace(4, 8, 1000)
# x2_plot = (-log_reg.coef_[0] * x1_plot - log_reg.interception_) / log_reg.coef_[1]
# plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color = "orange")
# plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color = "pink")
# plt.plot(x1_plot, x2_plot)
# plt.show()