__author__ = 'huangdaoxu'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def class_mean(x, y):
    classes = set(y)
    mean = []
    for cls in classes:
        mean.append(np.mean(x[y == cls,:], axis=0))
    return np.array(mean), classes

# 计算类内散度矩阵
def min_Sw(x, y):
    mean, classes = class_mean(x, y)
    Sw = np.zeros((x.shape[1], x.shape[1]))
    for cls, m in zip(classes, mean):
        Sw += np.dot((x[y == cls, :] - m).T, (x[y == cls, :] - m))
    return Sw

# 计算内间散度矩阵
def max_Sb(x, y):
    all_mean = np.mean(x, axis=0)
    mean, classes = class_mean(x, y)
    Sb = np.zeros((x.shape[1], x.shape[1]))
    for cls, m in zip(classes, mean):
        each_count = (y == cls).shape[0]
        Sb += each_count * np.dot((m - all_mean).T, (m - all_mean))
    return Sb

# 计算Sw^-1 * Sb的特征向量以及特征值，排序筛选出最大的n_components个特征值以及特征向量，
# 将特征向量组合成W，从而数据集点乘W得到降维后的数据
def lda(x, y, n_components=2):
    Sw = min_Sw(x, y)
    Sb = max_Sb(x, y)
    eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
    eig_pairs = [[np.abs(eig_vals[i]), eig_vecs[:, i]] for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    class_count = len(set(y))
    if n_components < class_count:
        if n_components==2:
            W = np.hstack((eig_pairs[0][1].reshape(x.shape[1], 1), eig_pairs[1][1].reshape(x.shape[1], 1)))
        else:
            W = ...
    else:
        raise ValueError('n_components must less than class count.')
    return W


from sklearn.datasets import load_iris
data_set = load_iris()
X,y = data_set.data,data_set.target

w = lda(X, y, n_components=2)
new_X = np.abs(X.dot(w))
fig = plt.figure()
plt.scatter(new_X[:, 0], new_X[:, 1],marker='o',c=y)
plt.show()

# 对比sklearn中实现的基于特征值与特征向量的LDA线性分析方法，经过源码比较后发现多分类实现有点区别。
# sklearn通过全局散度矩阵减去类内散度矩阵来求得类间散度矩阵，sklearn的实现更
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver='eigen', n_components=2)
lda.fit(X,y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.show()