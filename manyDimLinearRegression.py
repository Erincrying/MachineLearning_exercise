# 多元线性回归问题；解析解+梯度下降法（也能解决一元线性回归问题）
import numpy as np
from sklearn import datasets
from sklearn.metrics import r2_score
# from .metrics import r2_score

class LinearRegression:
  def __init__(self):
      # 初始化Linear Regression模型
      self.coef_ = None # 系数，对应theta1-n，对应的向量
      self.interception_ = None # 截距，对应theta0
      self._theta = None # 定义私有变量，整体计算的theta



  '''
    梯度下降
  '''
  def fit_gd(self, X_train, y_train, eta = 0.01, n_iters = 1e4): 
    # 根据训练数据集X_train, y_ .train训练Linear Regression模型
    # X_train的样本数量和y_train的标记数量应该是一致的
    # 使用shape[0]读取矩阵第一维度的长度，在这里就是列数
    assert X_train.shape[0] == y_train.shape[0], \
    "the size of x_ .train must be equal to the size of y_ train"
    # 损失函数
    def J(theta, X_b, y):
      try:
        return np.sum((y - X_b.dot(theta)) ** 2 / len(y))
      except:
        return float('inf') # 返回float最大值

    # 梯度(比较笨的方法)
    def dJ(theta, X_b, y):
      res = np.empty(len(theta)) # 开一个和theta一样大的空间(因为要对theta的每一个元素求偏导)
      res[0] = np.sum(X_b.dot(theta) - y) # 第一行单独写，其他的要写一个循环
      for i in range(1, len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:,i]) # X_b[:,i]相当于取出第i列，也就是第i个特征值
      return res * 2 / len(X_b)


    # 求解theta矩阵
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
    解析解
  '''
  # 正规化方程
  # 训练过程（解析解）
  def fit_normal(self, X_train, y_train): # X_train和y_train都是矩阵
    # 根据训练数据集X_train, y_ .train训练Linear Regression模型
    # X_train的样本数量和y_train的标记数量应该是一致的
    # 使用shape[0]读取矩阵第一维度的长度，在这里就是列数
    assert X_train.shape[0] == y_train.shape[0], \
    "the size of x_ .train must be equal to the size of y_ train"
    # np.hstack():在水平方向上平铺，就是在横向上多加一列
    # np.ones(矩阵大小, 列数)是增加一列恒为1的一列
    # 得到X_b
    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    # X_b.T是X_b的转置,.dot是点乘,np.linalg.inv是求逆
    self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

    self.interception_ = self._theta[0] # 截距
    self.coef_ = self._theta[1:] # 系数
    return self
  
  '''
    预测过程
  '''
  def predict(self, X_predict):
    # 给定待预测数据集X_predict,返回表示X_predict的结果向量
    X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
    return X_b.dot(self._theta)

  '''
    确定精度，评价多元线性回归的结果
  '''
  def score(self, X_test, y_test):
    # 根据测试数据集X_test 和y_test 确定当前模型的准确度
    y_predict = self.predict(X_test)
    return r2_score(y_test, y_predict) # r2_score求真值y_test和预测值y_predict的r方

  '''
    显示属性
  '''
  def __repr__(self):
      return "LinearRegression()"


# 加载波士顿房价数据集，并划分为X_train,y_train
# 波士顿房价
boston = datasets.load_boston()
X_train = boston.data
y_train = boston.target

# 实例化(数学解解法)
reg = LinearRegression()
# 进行训练
reg.fit_normal(X_train, y_train) # LinearRegression()
# 截距
print('数学解系数reg.coef_', reg.coef_)
print('数学解截距reg.interception_', reg.interception_)
reg.predict([X_train[0]])
print('数学解预测值', reg.predict([X_train[0]]))

# 实例化(梯度下降解法)
lin_reg = LinearRegression()
# 设置随机种子
np.random.seed(666)
# 用一维数据方便可视化，但实际是对矩阵进行操作
x = np.random.random(size = 100)
y = x * 3. + 4. + np.random.normal(size = 100) # 噪音：均值为0，方差为1

X = x.reshape(-1, 1) # 数组array中的方法，作用是将数据重新组织成100行1列的数据
# 梯度下降法进行训练
lin_reg.fit_gd(X, y) # LinearRegression()

print('梯度下降系数reg.coef_', lin_reg.coef_)
print('梯度下降截距reg.interception_', lin_reg.interception_)

