import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import preprocessing
import datetime

# 通过python console查看维度
features = pd.read_csv('data/temps.csv')

# 看看数据长什么样子
# print(features.head())

# 对字符串特征做独热编码
features = pd.get_dummies(features)
print(features.head())

# 标签
labels = np.array(features['actual'])

# 在特征中去掉标签
features = features.drop('actual', axis=1)

# 名字单独保存一下，以备后患
feature_list = list(features.columns)
# 特征标准化处理
input_features = preprocessing.StandardScaler().fit_transform(features)
# 转为tensor格式
x = torch.tensor(input_features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.float)

# 权重参数初始化，一层hidden layer
weights = torch.randn((14, 128), dtype=torch.float, requires_grad=True)
biases = torch.randn(128, dtype=torch.float, requires_grad=True)
weights2 = torch.randn((128, 1), dtype=torch.float, requires_grad=True)
biases2 = torch.randn(1, dtype=torch.float, requires_grad=True)

learning_rate = 0.001
losses = []

# 简单的手写梯度下降
for i in range(1000):
    # 计算隐层，mm为矩阵乘法
    hidden = x.mm(weights) + biases
    # 加入激活函数
    hidden = torch.relu(hidden)
    # 预测结果
    predictions = hidden.mm(weights2) + biases2
    # 代价函数为均方误差
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())

    # 打印损失值
    if i % 100 == 0:
        print('loss:', loss)
    # 返向传播计算
    loss.backward()

    # 更新参数
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)

    # 每次迭代都得记得清空
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()