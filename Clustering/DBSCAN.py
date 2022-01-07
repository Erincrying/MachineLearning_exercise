'''
  密度聚类：DBSCAN
'''
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

x, y_true = datasets.make_blobs(n_samples=200, centers=2,cluster_std=0.60, random_state=0)
x2, y2_true = datasets.make_moons(n_samples=200, noise=0.07, random_state=0)
x3, y3_true = datasets.make_circles(n_samples=200, noise=0.07, random_state=0, factor=0.4)

# DBSCAN eps：领域大小，min_samples：邻域内最小样本数量
# gmm = cluster.DBSCAN()
gmm = cluster.DBSCAN(eps=0.2, min_samples=4)
label = gmm.fit_predict(x)
label2 = gmm.fit_predict(x2)
label3 = gmm.fit_predict(x3)


# 初始数据集
plt.title("blobs-initial")
plt.scatter(x[y_true == 0, 0], x[y_true == 0, 1], color = "orange")
plt.scatter(x[y_true == 1, 0], x[y_true == 1, 1], color = "pink")
plt.show()
plt.title("moons-initial")
plt.scatter(x2[y2_true == 0, 0], x2[y2_true == 0, 1], color = "orange")
plt.scatter(x2[y2_true == 1, 0], x2[y2_true == 1, 1], color = "pink")
plt.show()
plt.title("circles-initial")
plt.scatter(x3[y3_true == 0, 0], x3[y3_true == 0, 1], color = "orange")
plt.scatter(x3[y3_true == 1, 0], x3[y3_true == 1, 1], color = "pink")
plt.show()


plt.scatter(x[:, 0], x[:, 1], c=label)
plt.title("blobs-result")
plt.show()

plt.scatter(x2[:, 0], x2[:, 1], c=label2)
plt.title("moons-result")
plt.show()

plt.scatter(x3[:, 0], x3[:, 1], c=label3)
plt.title("circles-result")
plt.show()
