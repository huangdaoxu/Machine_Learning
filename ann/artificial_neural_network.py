# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:45:30 2017

@author: huangdaoxu
"""

import numpy as np

#简单三层神经网络实现
class NeuralNetworkClassifer(object):
    
    def __init__(self,NeuronsNum,FeatureLen,Learn_rate,C):
        """
        NeuronsNum: neuron size(int)
        FeatureLen: feature‘s dimension(int)
        Learn_rate: learning rate(float)
        C         : to prevent over-fitting(float)
        """
        self.NeuronsNum = NeuronsNum
        self.FeatureLen = FeatureLen
        self.Learn_rate = Learn_rate
        self.C = C
        #第一层传导到第二层的权重值矩阵初始化
        self.Weights12 = np.random.normal(loc=0, scale=1,size = [self.FeatureLen,self.NeuronsNum])
        #第一层传导到第二层的偏置项初始化
        self.Bias12 = np.random.normal(loc=0, scale=1,size = [1,self.NeuronsNum])
        #第二层传导到第三层的权重值矩阵初始化
        self.Weights23 = np.random.normal(loc=0, scale=1,size = [self.NeuronsNum,1])
        #第二层传导到第三层的偏置项初始化
        self.Bias23 = np.random.normal(loc=0, scale=1,size = 1)

    #激活函数
    def __sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def __relu(self,x):
        return np.maximum(0.0, x)
        
    def __tanh(self,x):
        return np.tanh(x)
    #训练
    def train(self,feature,label):
        #将feature转化为np.array
        feature = np.array(feature)
        result = self.predict(feature)
        self.__gradient_descent_sigmoid(label,result[0])
    
    #常用全连接层后输出函数          
    def __softmax(self,x):
        tmp = np.exp(x - np.max(x))
        return tmp / tmp.sum()
    
    def __gradient_descent_sigmoid(self,label,label_):
        #输出层残差
        output_error = [0.0] * len(label)
        #output_error = label_*(1-label_)*(label-label_)
        for i in xrange(len(label)):
            output_error[i] = label_[i] * (1-label_[i]) * (label[i] - label_[i])
        #隐藏层残差
        #未完待续
    
    def predict(self,feature):
        #输入层前向传导到第二层的计算结果
        result12 = self.__relu(np.dot(feature,self.Weights12) + self.Bias12)
        #第二层到第三层的计算结果
        result23 = self.__softmax(np.dot(result12,self.Weights23) + self.Bias23)
        return result23
