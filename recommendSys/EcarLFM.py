# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 16:09:09 2017

@author: hdx
"""

import pandas as pd
import numpy as np
import random


class LatentFactorModel(object):
    def __init__(self, ratio, f, learning_rate, c, iteration_counts):
        """
        :param ratio: 正负样本比例
        :param f: 隐特征个数
        :param learning_rate: 学习率
        :param c: 正则化参数
        :param iteration_counts : 迭代次数
        """
        self._ratio = ratio
        self._f = f
        self._learning_rate = learning_rate
        self._c = c
        self._iteration_counts = iteration_counts

    def _getData(self):
        """
        :return: 商品浏览记录总数据
        """
        data = [('a',101,1),('a',111,1),('a',141,0), 
                ('b',111,0),('b',151,1),('b',131,0), 
                ('c',121,1),('c',161,0),('c',141,0), 
                ('d',111,1),('d',161,1),('d',141,0),('d',121,0), 
                ('e',131,1),('e',151,0),('e',171,0),
                ('f',181,0),('f',191,1),
                ('g',101,1),('g',201,0)]
        data = pd.DataFrame(np.array(data))
        data.columns = ['openid', 'productid', 'status']
        return data

    def _getUserPositiveItem(self, data, openid):
        """
        :param data: 总数据
        :param openid: 用户id
        :return: 正反馈物品
        """
        positiveItemList = data[data['openid'] == openid]['productid'].unique().tolist()
        return positiveItemList

    def _getUserNegativeItem(self, data, openid):
        """
        :param data: 总数据
        :param openid: 用户id
        :return: 未产生行为的物品
        """
        otherItemList = list(set(data['productid'].unique().tolist()) - set(self._getUserPositiveItem(data, openid)))
        negativeItemList = random.sample(otherItemList, self._ratio * len(self._getUserPositiveItem(data, openid)))
        return negativeItemList

    def _initUserItem(self, data):
        userItem = {}
        for openid in data['openid'].unique():
            positiveItem = self._getUserPositiveItem(data, openid)
            negativeItem = self._getUserNegativeItem(data, openid)
            itemDict = {}
            for item in positiveItem: itemDict[item] = 1
            for item in negativeItem: itemDict[item] = 0
            userItem[openid] = itemDict
        return userItem


    def _initParams(self, data):
        """
        初始化参数
        :param data: 总数据
        :return: p\q参数
        """
        p = np.random.rand(len(data['openid'].unique().tolist()), self._f)
        q = np.random.rand(self._f, len(data['productid'].unique().tolist()))
        userItem = self._initUserItem(data)
        p = pd.DataFrame(p, columns=range(0, self._f), index=data['openid'].unique().tolist())
        q = pd.DataFrame(q, columns=data['productid'].unique().tolist(), index=range(0, self._f))
        return p, q, userItem

    def sigmod(self, x):
        """
        单位阶跃函数,将兴趣度限定在[0,1]范围内
        :param x: 兴趣度
        :return: 兴趣度
        """
        y = 1.0 / (1 + np.exp(-x))
        return y

    def predict(self, p, q, openid, productid):
        """
        利用参数p, q预测目标用户对目标物品的兴趣度
        """
        p = np.mat(p.ix[openid].values)
        q = np.mat(q[productid].values).T
        r = (p * q).sum()
        r = self.sigmod(r)
        return r

    def train(self):
        data = self._getData()
        p, q, userItem = self._initParams(data)

        for step in xrange(1, self._iteration_counts+1):
            for openid, samples in userItem.items():
                for productid, r in samples:
                    loss = r - self.predict(p, q, openid, productid)
                    for f in xrange(0, self._f):
                        print('step %d openid %s class %d loss %f' % (step, openid, f, np.abs(loss)))
                        p[f][openid] += self._learning_rate * (loss * q[productid][f] - self._c * p[f][openid])
                        q[productid][f] += self._learning_rate * (loss * p[f][openid] - self._c * q[productid][f])
            if step % 5 == 0:
                self._learning_rate *= 0.9
        return p, q, data

    def recommend(self, data, p, q):
        rank = []
        for openid in data['openid'].unique():
            for productid in data['productid'].unique():
                rank.append((openid, productid, self.predict(p, q, openid, productid)))
        return rank




