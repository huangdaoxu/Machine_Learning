__author__ = 'huangdaoxu'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class LDA(object):
    def __init__(self, n_components=2):
        self.n_components=n_components

    def _class_mean(self, x, y):
        classes = set(y)
        mean = []
        for cls in classes:
            mean.append(np.mean(x[y == cls, :], axis=0))
        return np.array(mean), classes

    # 计算类内散度矩阵
    def _min_Sw(self, x, y):
        mean, classes = self._class_mean(x, y)
        Sw = np.zeros((x.shape[1], x.shape[1]))
        for cls, m in zip(classes, mean):
            Sw += np.cov((x[y == cls, :]).T, bias=True)
        Sw /= len(classes)
        return Sw

    # 计算内间散度矩阵, 多分类使用St-Sw得到
    def _max_Sb(self, x, y):
        all_mean = np.mean(x, axis=0)
        mean, classes = self._class_mean(x, y)
        Sb = np.zeros((x.shape[1], x.shape[1]))
        for cls, m in zip(classes, mean):
            each_count = (y == cls).shape[0]
            Sb += each_count * np.dot((m - all_mean).T, (m - all_mean))
        return Sb

    # 计算Sw^-1 * Sb的特征向量以及特征值，排序筛选出最大的n_components个特征值以及特征向量，
    # 将特征向量组合成，从而数据集点乘特征向量得到数据
    def _lda(self, x, y):
        Sw = self._min_Sw(x, y)
        St = np.cov(x.T, bias=True)
        Sb = St - Sw
        eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        return eig_vecs

    def transform(self, x, y):
        class_count = len(set(y))
        if self.n_components < class_count:
            w = self._lda(x, y)
            new_X = np.dot(X, w)[:, :self.n_components]
        else:
            raise ValueError('n_components must less than class count.')
        return new_X


if __name__ == '__main__':
    data_set = load_iris()
    X, y = data_set.data, data_set.target
    lda = LDA(n_components=2)
    new_X = lda.transform(X, y)
    plt.scatter(new_X[:, 0], new_X[:, 1],marker='o',c=y)
    plt.show()