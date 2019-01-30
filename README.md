# 机器学习实战

ann使用（python 2.7.13 | numpy1.10）实现的简单神经网络模型。该模型中未加入正则化。

ocr_cnn_tensorflow使用（python 3.6 | tensorflow 1.8）编写的4位固定长度数字验证码识别模型。4万张照片迭代50次，准确率达到97%以上。

ocr_lstmctc_tensorflow使用（python 3.6 | tensorflow 1.8）实现的lstm+ctc进行变长验证码识别的模型，4万张图片迭代50次，准确率达到95%以上。

recommendSys 是一个以numpy为基础编写的推荐系统，分别为基于用户的协同过滤、基于物品的协同过滤、LFM模型，以隐性反馈数据为基础（显性反馈数据同样适用这些模型）。LFM模型说明链接如下：http://www.jianshu.com/p/9270edf4b08b 。

autoencoder使用（python 2.7.13 | tensorflow version1.2）实现的自编码模型，该模型常用于提取非线性有效特征以及降低特征维度（作用与PCA降维类似），激活函数的选取非常重要，在不同的特征集上需要找到最合适的参数。

Linear Discriminant Analysis使用（python3.6 | numpy1.14）实现的线性判别分析模型，核心通过特征值分解实现，而sklearn中有三种kernel。它属于监督学习方式的特征降维，而PCA是非监督学习。理论知识及公式基于周志华老师的机器学习书籍。

Principal components analysis使用（python3.6 | numpy1.14）实现的主成分分析模型，核心通过特征值分解与奇异值分解实现。

abnormal_url_detection（python3.6 | tensorflow 1.10 | gensim）异常url检测（web攻击检测）lstm+dynamic_rnn+gensim中word2vec。
