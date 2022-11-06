import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')
features = pd.read_csv('data/temps.csv')

features.head()
# one-hot编码
features = pd.get_dummies(features)
print(features.head(5))

# 标签
labels = np.array(features['actual'])
# 去掉标签列
features = features.drop('actual', axis=1)
# 列名字单独保存一下
features_list = list(features.columns)
# 转换格式
features = np.array(features)

from sklearn import preprocessing

# 数据归一化
input_features = preprocessing.StandardScaler().fit_transform(features)
x = torch.tensor(input_features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.float)
#
# 权重参数初始化
weights = torch.randn((14, 128), dtype=torch.float, requires_grad=True)
biases = torch.randn(128, dtype=torch.float, requires_grad=True)
weights2 = torch.randn((128, 1), dtype=torch.float, requires_grad=True)
biases2 = torch.randn(1, dtype=torch.float, requires_grad=True)
#
learning_rate = 0.001
losses = []
#
for i in range(1000):
    hidden = x.mm(weights) + biases

    hidden = torch.relu(hidden)

    predictions = hidden.mm(weights2) + biases2

    loss = torch.mean((predictions - y) ** 2)

    losses.append(loss.data.numpy())

    if i % 100 == 0:
        print(f"loss:{loss}")
    # 反向传播
    loss.backward()
    # 更新参数
    weights.data.add_(-learning_rate*weights.grad.data)
    biases.data.add_(-learning_rate*biases.grad.data)
    weights2.data.add_(-learning_rate*weights2.grad.data)
    biases2.data.add_(-learning_rate*biases2.grad.data)

    # 清空梯度
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()
