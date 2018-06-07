__author__ = 'huangdaoxu'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

class PCA(object):
    """
    n_components: transformed decomposition
    solver: kernel algorithm
    """
    def __init__(self, n_components=2, solver='svd'):
        self.n_components=n_components
        self.solver=solver

    # 使用特征值分解方式降维
    def _eig_solver(self, x):
        covs = np.cov(x.T, bias=True)
        eig_vals, eig_vecs = np.linalg.eig(covs)
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        new_X = np.dot(x, eig_vecs)[:, :self.n_components]
        return new_X

    # 使用svd奇异值分解方式降维
    def _svd_solver(self, x):
        u, s, v = np.linalg.svd(x)
        new_X = np.dot(u[:, :self.n_components] * s[:self.n_components], v[:self.n_components, :])
        return new_X[:,:self.n_components]

    def transform(self, x):
        if self.n_components >= x.shape[1]:
            raise ValueError('n_components must less than class count.')

        if self.solver == 'eigen':
            new_X = self._eig_solver(x)
        elif self.solver == 'svd':
            new_X = self._svd_solver(x)
        else:
            raise ValueError('solver is not exist.')
        return new_X


if __name__ == '__main__':
    data_set = load_iris()
    X, y = data_set.data, data_set.target
    pca = PCA(n_components=2, solver='svd')
    new_X = pca.transform(X)
    plt.scatter(new_X[:, 0], new_X[:, 1], marker='o', c=y)
    plt.show()