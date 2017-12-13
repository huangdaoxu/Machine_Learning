# -*- coding: utf-8 -*-

import EcarItemCF
import EcarUserCF
import EcarLFM

if __name__ == "__main__":
    # 特征1为用户id，特征2为产品id，特征3为是否已经购买产品的标示，
    # 目前特征3作用不大，因为只需要浏览过商品，我们就认为用户对产品是感兴趣的
    data = [('a',101,1),('a',111,1),('a',141,0),
                ('b',111,0),('b',151,1),('b',131,0),
                ('c',121,1),('c',161,0),('c',141,0),
                ('d',111,1),('d',161,1),('d',141,0),('d',121,0),
                ('e',131,1),('e',151,0),('e',171,0),
                ('f',181,0),('f',191,1),
                ('g',101,1),('g',201,0)]
    usercf = EcarUserCF.recommendation(3, data)
    print 'usercf result:\n{0}'.format(usercf)

    itemcf = EcarItemCF.recommendation(3, data)
    print 'itemcf result:\n{0}'.format(itemcf)

    lfm = EcarLFM.LatentFactorModel(1, 5, 0.5, 0.01, 10)
    p, q, data = lfm.train(data)
    print 'lfm result:\n{0}'.format(lfm.recommend(data, p, q))