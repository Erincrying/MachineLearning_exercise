# 模拟梯度下降
import numpy as np
import matplotlib.pyplot as plt

# 损失函数
plot_x = np.linspace(-1, 6, 141) 
plot_y = (plot_x - 2.5) ** 2 - 1 # 先自己定义了一个二次函数

# 绘制
plt.plot(plot_x, plot_y)
plt.show()

# 对损失函数求导
def dEwb(theta):
    return 2 * (theta -2.5)

# 损失函数
def E_function(theta):
    return (theta - 2.5) ** 2 - 1

theta = 0.0
α = 0.1 # 设置阿尔法
epsilon = 1e-8 # 差值精度，确定最优解点
history_theta = [ theta ] # 记录theta值用于之后的对比
# 找梯度最小的情况
while True:
    gradient = dEwb(theta) # 求精度
    last_theta = theta # theta重新赋值前，记录上一场的值
    theta = theta - α * gradient # 通过一定的α取得下一个点的theta
    history_theta.append(theta) # 更新history_theta
    # 最近两点的损失函数差值小于一定精度，退出循环
    if(abs(E_function(theta) - E_function(last_theta)) < epsilon):
        break

plt.plot(plot_x, E_function(plot_x))
plt.plot(np.array(history_theta), E_function(np.array(history_theta)), color = 'r', marker = '+')
plt.show()
