"""
Created on Fri Jul 21 11:45:30 2017

@author: huangdaoxu
"""

import numpy as np

# 3 layers nueral network
class NeuralNetworkClassifer(object):
    
    def __init__(self, NeuronsNum, FeatureLen, Learn_rate):
        """
        NeuronsNum: neuron size(int)
        FeatureLen: feature‘s dimension(int)
        Learn_rate: learning rate(float)
        """
        self.NeuronsNum = NeuronsNum
        self.FeatureLen = FeatureLen
        self.Learn_rate = Learn_rate
        # 第一层传导到第二层的权重值矩阵初始化
        self.Weights12 = np.random.normal(loc=0, scale=1, size=[self.FeatureLen, self.NeuronsNum])
        # 第一层传导到第二层的偏置项初始化
        self.Bias12 = np.random.normal(loc=0, scale=1, size=[1, self.NeuronsNum])
        # 第二层传导到第三层的权重值矩阵初始化
        self.Weights23 = np.random.normal(loc=0, scale=1, size=[self.NeuronsNum, 1])
        # 第二层传导到第三层的偏置项初始化
        self.Bias23 = np.random.normal(loc=0, scale=1, size=1)

    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def __relu(self, x):
        return np.maximum(0.0, x)
        
    def __tanh(self, x):
        return np.tanh(x)

    # derivative of sigmoid
    def __dsigmoid(self, x):
        return x * (1 - x)

    def fit(self, feature, label):
        #将feature转化为np.array
        feature = np.array(feature)
        # 输入层前向传导到第二层的计算结果
        result12 = self.__sigmoid(np.dot(feature, self.Weights12) + self.Bias12)
        # 第二层到第三层的计算结果
        result23 = self.__sigmoid(np.dot(result12, self.Weights23) + self.Bias23)
        self.__gradient_descent_sigmoid(feature, label, result12, result23)

    def __softmax(self, x):
        tmp = np.exp(x - np.max(x))
        return tmp / tmp.sum()

    def __gradient_descent_sigmoid(self, x, label, result12, result23):
        # 损失函数定义为 最小二乘法1/2*(label - result)^2
        error23 = label - result23
        delta23 = error23 * self.__dsigmoid(result23)
        error12 = np.dot(delta23, self.Weights23.T)
        delta12 = error12 * self.__dsigmoid(result12)
        self.Weights23 += self.Learn_rate * np.dot(result12.T, delta23)
        self.Weights12 += self.Learn_rate * np.dot(x.T, delta12)

        self.Bias23 += self.Learn_rate * np.mean(delta23, axis=0)
        self.Bias12 += self.Learn_rate * np.mean(delta12, axis=0)

        print("loss: ", np.mean(np.square(error23)))


    def predict(self,feature):
        # 输入层前向传导到第二层的计算结果
        result12 = self.__sigmoid(np.dot(feature,self.Weights12) + self.Bias12)
        # 第二层到第三层的计算结果
        result23 = self.__sigmoid(np.dot(result12,self.Weights23) + self.Bias23)
        return result23
