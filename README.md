# 机器学习实战

ocr_cnn_tensorflow 为使用（python 2.7.13 | tensorflow version1.2）编写的4位固定长度数字验证码识别模型，本人训练的模型保存在了save文件夹中。可直接restore后使用（如不能使用，请重新训练），大约4万张照片迭代50次，准确率达到97%以上。

ocr_lstmctc_tensorflow 为使用（python 2.7.14 | tensorflow version1.4）实现的一个lstm+ctc进行变长验证码识别的模型，目前训练后损失还未能降下来，可能是网络模型的问题，后期修复后更新代码。

recommendSys 是一个以numpy为基础编写的推荐系统，分别为基于用户的协同过滤、基于物品的协同过滤、LFM模型。分别以隐性反馈数据为基础（显性反馈数据同样适用这些模型）。LFM模型说明链接如下：http://www.jianshu.com/p/9270edf4b08b 。