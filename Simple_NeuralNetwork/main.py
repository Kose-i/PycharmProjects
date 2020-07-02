import sys, os
sys.path.append(os.pardir)
import numpy as np
from keras.datasets.mnist import load_data
#from gradient_simplenet
from two_layer_net import TwoLayerNet
from keras.utils import np_utils

if __name__=='__main__':
    (x_train, t_train), (x_test, t_test) = load_data() # mnist-data
    t_train = np_utils.to_categorical(t_train, 10).astype('float32')
    t_test  = np_utils.to_categorical(t_test,  10).astype('float32')
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    train_size = x_train.shape[0]

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # hyper parameters
    iters_num = 10000
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.numerical_gradient(x_batch, t_batch)
        #grad = network.gradient(x_batch, t_batch)
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
    print(train_loss_list)

    #print("accuracy:", network.accuracy(x_test_reshape, t_test))
