import numpy as np
import matplotlib.pyplot as plt

# x轴数据
x_data = np.arange(20)
# y轴数据
y_data = np.array([0.4, 0.8, 1.1, 2.1, 2.8, 2.7, 3.5, 4.6, 5.1, 4.5, 6.0, 5.5, 6.9, 6.8, 7.6, 8.0, 8.8, 8.5, 9.5, 9.3])
print(x_data)
print(y_data)

plt.scatter(x_data, y_data)
plt.show()

def compute_mse(b, w, x_data, y_data):
    """
    求均方差
    :param b: 截距
    :param w: 斜率
    :param x_data: 特征数据
    :param y_data: 标签数据
    """
    
    total_error = 0.0
    for i in range(0, len(x_data)):
        total_error += (y_data[i] - (b + w * x_data[i])) ** 2
    return total_error / len(x_data)

def run_gradient_descent(x_data, y_data, b, w, learn_rate, epochs):
    """
    运行梯度下降
    :param x_data: 待训练的特征数据
    :param y_data: 标签数据
    :param b: 截距
    :param w: 斜率
    :param learn_rate: 学习曲率
    :param epochs: 训练次数
    """
    
    m = float(len(x_data))
    for i in range(epochs):
        b_grad = 0 # 损失（代价）函数对b的梯度
        w_grad = 0 # 损失（代价）函数对w的梯度
        for j in range(0, len(x_data)):
            b_grad += (1/m) * ((b + w * x_data[j]) - y_data[j])
            w_grad += (1/m) * ((b + w * x_data[j]) - y_data[j]) * x_data[j]
        # 根据梯度和学习曲率修正截距b和斜率w
        b -= learn_rate * b_grad
        w -= learn_rate * w_grad
        if i % 50 == 0:
        	# 每50次作图一次
            print("epochs:", i)
            plt.plot(x_data, y_data, "b.")
            plt.plot(x_data, b + w * x_data, "r")
            plt.show()
            print("mse:", compute_mse(b, w, x_data, y_data))
            print("------------------------------------------------------------------------------------------------------------")
    return b, w
learn_rate = 0.0001 # 学习曲率
b = 0 # 截距
w = 0 # 斜率
epochs = 500 # 批次数

print("Start args: b = {0}, w = {1}, mse= {2}".format(b, w, compute_mse(b, w, x_data, y_data)))
print("Running...")
b, w = run_gradient_descent(x_data, y_data, b, w, learn_rate, epochs)
print("Finish args: iterations = {0}  b = {1}, w = {2}, mse= {3}".format(epochs, b, w, compute_mse(b, w, x_data, y_data)))

plt.plot(x_data, y_data, "b.")
plt.plot(x_data, b + w * x_data, "r")
plt.show()

