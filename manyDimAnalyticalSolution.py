# 多元线性回归问题；解析解（也能解决一元线性回归问题）
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

  # 正规化方程
  # 训练过程
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
  
  # 预测过程
  def predict(self, X_predict):
    # 给定待预测数据集X_predict,返回表示X_predict的结果向量
    X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
    return X_b.dot(self._theta)

  # 确定精度，评价多元线性回归的结果
  def score(self, X_test, y_test):
    # 根据测试数据集X_test 和y_test 确定当前模型的准确度
    y_predict = self.predict(X_test)
    return r2_score(y_test, y_predict) # r2_score求真值y_test和预测值y_predict的r方

  # 显示属性
  def __repr__(self):
      return "LinearRegression()"


# 加载波士顿房价数据集，并划分为X_train,y_train
# 波士顿房价
boston = datasets.load_boston()
X_train = boston.data
y_train = boston.target

# 实例化
reg = LinearRegression()
# 进行训练
reg.fit_normal(X_train, y_train) # LinearRegression()
# 截距
print('系数reg.coef_', reg.coef_)
print('截距reg.interception_', reg.interception_)
reg.predict([X_train[0]])
print('预测值', reg.predict([X_train[0]]))
