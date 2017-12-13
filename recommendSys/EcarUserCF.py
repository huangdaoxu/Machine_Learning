# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 16:09:09 2017

@author: hdx
"""

import pandas as pd
import numpy as np
from operator import itemgetter


def user_similarity(data):
    data = pd.DataFrame(np.array(data))
    data.columns = ['openid', 'productid', 'status']
    openid_list = data['openid'].unique().tolist()
    # 用户相似矩阵
    w = []
    for x in openid_list:
        row_matrix = []
        x_set = set(data[data['openid'] == x]['productid'].unique().tolist())
        for y in openid_list:
            if x == y:
                row_matrix.append(0)
            else:
                y_set = set(data[data['openid'] == y]['productid'].unique().tolist())
                row_matrix.append(len(x_set & y_set)/np.sqrt(len(x_set) * len(y_set)))

        w.append(row_matrix)

    return np.array(w), openid_list, data


def recommendation(k, data):
    """
    :param k: top k
    :param data: input
    :return:user and product relationship
    """
    w, openid_list, data = user_similarity(data)
    rank = []
    for i in xrange(len(openid_list)):
        for product_id in data['productid'].unique():
            tmp_list = [w[i], openid_list]
            top_k_similarity = sorted(map(list, zip(*tmp_list)), key=itemgetter(0), reverse=True)[0:k]
            P = 0
            R = 1
            for similar_user in top_k_similarity:
                if product_id not in data[data['openid'] == similar_user[1]]['productid'].values:
                    continue
                P += R * similar_user[0]
            rank.append((openid_list[i], product_id, P))
    return rank


