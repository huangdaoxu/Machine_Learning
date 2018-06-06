# 机器学习实战

ann 是未使用深度学习框架实现的简单神经网络模型，使用基础库为numpy。不过该模型中未加入防止过拟合如正则化等，后期可能会在目前版本上加上正则化。

ocr_cnn_tensorflow 为使用（python 2.7.13 | tensorflow version1.2）编写的4位固定长度数字验证码识别模型，本人训练的模型保存在了save文件夹中。可直接restore后使用（如不能使用，请重新训练），大约4万张照片迭代50次，准确率达到97%以上。

ocr_lstmctc_tensorflow 为使用（python 2.7.14 | tensorflow version1.4）实现的一个lstm+ctc进行变长验证码识别的模型，目前训练后损失还未能降下来，可能是网络模型的问题，后期修复后更新代码。

recommendSys 是一个以numpy为基础编写的推荐系统，分别为基于用户的协同过滤、基于物品的协同过滤、LFM模型，以隐性反馈数据为基础（显性反馈数据同样适用这些模型）。LFM模型说明链接如下：http://www.jianshu.com/p/9270edf4b08b 。

autoencoder为自编码模型，它属于非监督学习，该模型常用于提取有效特征以及降低特征维度（作用与PCA降维类似），需要注意的一点是，对于激活函数的选取非常重要，在不同的特征集上需要找到最合适的激活函数。

Linear Discriminant Analysis使用（python3.6 | numpy1.14）环境编写的线性判别分析模型，核心为通过特征值与特征向量来做变换，而sklearn中有三种kernel。它属于监督学习方式的特征降维，而PCA是非监督学习。