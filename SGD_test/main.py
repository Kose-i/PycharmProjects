from common.two_layer_net import TwoLayerNet
from keras.datasets.mnist import load_data
from keras.utils import np_utils
import numpy as np

#from SGD_class import SGD
#from Momentum_class import Momentum
from AdaGrad_class import AdaGrad

if __name__=='__main__':

    (x_train, t_train), (x_test, t_test) = load_data()
    x_train = x_train.reshape(x_train.shape[0], 784).astype('float32')/255.0
    x_test = x_test.reshape(x_test.shape[0], 784).astype('float32')/255.0
    t_train = np_utils.to_categorical(t_train, 10).astype('float32')
    t_test = np_utils.to_categorical(t_test, 10).astype('float32')
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    #optimizer = SGD()
    #optimizer = Momentum()
    optimizer = AdaGrad()

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(10000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        grads = network.gradient(x_batch, t_batch)
        params = network.params
        optimizer.update(params, grads)
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)
