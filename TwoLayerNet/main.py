# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from keras.datasets.mnist import load_data
from two_layer_net import TwoLayerNet

from keras.utils import np_utils

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_data()
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32')/255.0
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32')/255.0
t_train = np_utils.to_categorical(t_train, 10).astype('float32')
t_test = np_utils.to_categorical(t_test, 10).astype('float32')

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
