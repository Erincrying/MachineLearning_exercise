'''
  k均值算法：K-means
'''
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

# 4类簇
x, y_true = datasets.make_blobs(n_samples=200, centers=4,cluster_std=0.60, random_state=0)
x2, y2_true = datasets.make_moons(n_samples=200, noise=0.1, random_state=0)
x3, y3_true = datasets.make_circles(n_samples=200, noise=0.1, random_state=0, factor=0.4)


# KMeans
gmm = cluster.KMeans(2)
label = gmm.fit_predict(x)
label2 = gmm.fit_predict(x2)
label3 = gmm.fit_predict(x3)

print(y_true, 'make_blobs-label真实标签')
# print(y2_true, 'make_moons-label真实标签')
# print(label3, 'make_circles-label真实标签')

# print(label, 'make_blobs-label预测标签')
# print(label2, 'make_moons-label预测标签')
# print(label3, 'make_circles-label预测标签')



# 初始数据集（如果是4类簇）
plt.scatter(x[y_true == 0, 0], x[y_true == 0, 1], color = "orange")
plt.scatter(x[y_true == 1, 0], x[y_true == 1, 1], color = "pink")
plt.scatter(x[y_true == 2, 0], x[y_true == 2, 1], color = "red")
plt.scatter(x[y_true == 3, 0], x[y_true == 3, 1], color = "blue")
plt.show()
plt.scatter(x2[y2_true == 0, 0], x2[y2_true == 0, 1], color = "orange")
plt.scatter(x2[y2_true == 1, 0], x2[y2_true == 1, 1], color = "pink")
plt.show()
plt.scatter(x3[y3_true == 0, 0], x3[y3_true == 0, 1], color = "orange")
plt.scatter(x3[y3_true == 1, 0], x3[y3_true == 1, 1], color = "pink")
plt.show()


# plt.subplot(2, 2, 1)
# x[m,n]:m代表第m维，n代表m维中取第几段特征数据,x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据,
plt.scatter(x[:, 0], x[:, 1], c=label)
plt.title("blobs")
# plt.axis('off')
plt.show()

# plt.subplot(2, 2, 2)
plt.scatter(x2[:, 0], x2[:, 1], c=label2)
plt.title("moons")
# plt.axis('off')
plt.show()

# plt.subplot(2, 2, 3)
plt.scatter(x3[:, 0], x3[:, 1], c=label3)
plt.title("circles")
# plt.axis('off')
plt.show()
