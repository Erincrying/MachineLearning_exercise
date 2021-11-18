# 绘制信息熵（针对二分类问题）

import numpy as np
import matplotlib.pyplot as plt

'''
  计算信息熵
'''
def entropy(p):
  # 这里的p不仅可以是数字，也可以是向量或者是数组
  return -p * np.log(p) - (1-p) * np.log(1-p)


# x是向量，(0,1)均匀取200个值，不取特殊情况的0、1
x = np.linspace(0.01, 0.99, 200)
plt.plot(x, entropy(x))
plt.show()