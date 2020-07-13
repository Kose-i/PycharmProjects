# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data
from SimpleNetConvNetClass import SimpleConvNet
from common.trainer import Trainer
from keras.utils import np_utils

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_data()
x_train = x_train.astype('float32').reshape(x_train.shape[0],1,28,28)/255.0
x_test = x_test.astype('float32').reshape(x_test.shape[0],1,28,28)/255.0
t_train = np_utils.to_categorical(t_train, 10).astype('float32')
t_test = np_utils.to_categorical(t_test, 10).astype('float32')


# 処理に時間のかかる場合はデータを削減
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
