# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 16:09:09 2017

@author: hdx
"""

import pandas as pd
import numpy as np
from operator import itemgetter


def item_similarity():
    data = [('a',101,1),('a',111,1),('a',141,0), 
                ('b',111,0),('b',151,1),('b',131,0), 
                ('c',121,1),('c',161,0),('c',141,0), 
                ('d',111,1),('d',161,1),('d',141,0),('d',121,0), 
                ('e',131,1),('e',151,0),('e',171,0),
                ('f',181,0),('f',191,1),
                ('g',101,1),('g',201,0)]
    data = pd.DataFrame(np.array(data))
    data.columns = ['openid', 'productid', 'status']
    data['status'] = data['status'].astype(int)
    product_list = data['productid'].unique().tolist()
    # 相似度矩阵
    W = []
    for x in product_list:
        # 一行一行插入
        row_matrix = []
        x_set = set(data[data['productid'] == x]['openid'].unique().tolist())
        for y in product_list:
            if x == y:
                row_matrix.append(0)
            else:
                y_set = set(data[data['productid'] == y]['openid'].unique().tolist())
                row_matrix.append(len(x_set & y_set)/np.sqrt(len(x_set) * len(y_set)))

        W.append(row_matrix)
    return np.array(W), product_list, data


def recommendation(K):
    W, product_list, data = item_similarity()
    rank = []
    for openid in data['openid'].unique():
        for i in xrange(len(product_list)):
            tmp_list = [W[i], product_list]
            top_k_similarity = sorted(map(list, zip(*tmp_list)), key=itemgetter(0), reverse=True)[0:K]
            # 获取用户u对所有物品的兴趣度，只要有过浏览行为就认为对物品感兴趣
            Puj = 0
            Ru = 1
            for similar_good in top_k_similarity:
                if similar_good[1] not in data[(data['openid'] == openid)]['productid'].values:
                    continue
                Puj += Ru * similar_good[0]
            rank.append((openid, product_list[i], Puj))
    return rank

