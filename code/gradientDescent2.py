# 实现一元线性回归：梯度下降法
import numpy as np
import matplotlib.pyplot as plt

# x轴数据
x_data = np.arange(20)
# y轴数据
# y_data = np.array([0.4, 0.8, 1.1, 2.1, 2.8, 2.7, 3.5, 4.6, 5.1, 4.5, 6.0, 5.5, 6.9, 6.8, 7.6, 8.0, 8.8, 8.5, 9.5, 9.3])
y_data = np.random.rand(1, 20)
plt.scatter(x_data, y_data)
plt.show()

# 梯度下降
# α: 学习曲率
# epochs: 训练次数
def run_gradient_descent(x_data, y_data, b, w, α, epochs):    
    m = float(len(x_data))
    for i in range(epochs):
        b_grad = 0 # 损失函数对b的梯度
        w_grad = 0 # 损失函数对w的梯度
        for j in range(0, len(x_data)):
            b_grad += (1/m) * ((b + w * x_data[j]) - y_data[j])
            w_grad += (1/m) * ((b + w * x_data[j]) - y_data[j]) * x_data[j]
        # 根据梯度和学习曲率修正截距b和斜率w
        b -= α * b_grad
        w -= α * w_grad
        if i % 80 == 0:
        	# 每80次作图一次
            plt.scatter(x_data, y_data)
            plt.plot(x_data, b + w * x_data,  color = 'r')
            plt.show()
    return b, w

# 设置初始参数
α = 0.001 # 学习曲率
b = 0 # 截距
w = 0 # 斜率
epochs = 500 # 批次数

b, w = run_gradient_descent(x_data, y_data, b, w, α, epochs)
y_hat = w * x_data + b # 通过刚刚求解的w,b，基于每一个x所属的特征值进行预测
# 绘制
plt.scatter(x_data, y_data)
plt.plot(x_data, y_hat,  color = 'r')
plt.show()

