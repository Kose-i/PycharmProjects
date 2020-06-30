import sys, os
sys.path.append(os.pardir)
import numpy as np
from keras.datasets.mnist import load_data
#from gradient_simplenet
from two_layer_net import TwoLayerNet
from keras.utils import np_utils

if __name__=='__main__':
    (x_train, t_train), (x_test, t_test) = load_data() # mnist-data
    train_loss_list = []
    t_train = np_utils.to_categorical(t_train, 10)
    t_test  = np_utils.to_categorical(t_test,  10)

    # hyper parameters
    iters_num = 10000
    #iters_num = 1
    print(x_train.shape)
    train_size = x_train.shape[0]
    x_train_reshape = x_train.reshape(train_size, 784)
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    #print("accuracy:", network.accuracy(x_train_reshape, t_train))
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train_reshape[batch_mask].astype('float32') / 255.0

        t_batch = t_train[batch_mask].astype('float32')
        #print(x_batch.shape)
        #print(t_batch.shape)

        #grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)
        #print(grad)
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        loss = network.loss(x_batch, t_batch)
        print(loss)
        if len(train_loss_list) > 10 and train_loss_list[-1]<loss:
            #print(train_loss_list)
            #print("accuracy:", network.accuracy(x_test_reshape, t_test))
            exit()
        train_loss_list.append(loss)


    #print("accuracy:", network.accuracy(x_test_reshape, t_test))
