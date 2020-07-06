import sys, os
sys.path.append(os.pardir)
import numpy as np
from keras.datasets.mnist import load_data
from two_layer_net import TwoLayerNet
from keras.utils import np_utils


if __name__=='__main__':
    (x_train, t_train), (x_test, t_test) = load_data()
    x_train = x_train.reshape(x_train.shape[0], 784).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 784).astype('float32')
    t_train = np_utils.to_categorical(t_train, 10).astype('float32')
    t_test = np_utils.to_categorical(t_test, 10).astype('float32')

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = x_train[:3]
    t_batch = t_train[:3]
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key+":"+str(diff))
