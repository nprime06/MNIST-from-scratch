import sys
sys.path.append('../../Modules_old')
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from Dense import Dense
from ReLU import ReLU
from Reg import BatchNorm, Dropout

# Not yet implemented:
# momentum, decaying learning rate
# data aug

train_size = 60000

def softmax(x):
    x_max = np.max(x, axis = 1, keepdims = True)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x, axis = 1).reshape(-1, 1) + 1e-8)


class MLP():
    def __init__(self, hidden_sizes, epochs, batch_size, learning_rate, dropout_rate): # training is done in the class
        self.epochs = epochs
        self.batch_size = batch_size
        self.batches = train_size // self.batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        self.input_size = 784
        self.output_size = 10
        layer_sizes = [self.input_size] + hidden_sizes + [self.output_size]

        # architecture: Dense + (batchnorm) + relu
        self.layers = []
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], use_bias = False))
            #self.layers.append(BatchNorm())
            self.layers.append(ReLU())
            self.layers.append(Dropout(dropout_rate = self.dropout_rate))
        self.layers.append(Dense(layer_sizes[-2], layer_sizes[-1]))

        self.isTraining = True


    def forward(self, image):
        for layer in self.layers:
            image = layer.forward(image, isTraining = self.isTraining)

        return image


    def back(self, grad):
        for layer in reversed(self.layers):
            grad = layer.back(grad, learning_rate = self.learning_rate)

        return grad


    def train(self, x_train, y_train):
        self.isTraining = True

        for epoch in tqdm(range(self.epochs)):
            perm = np.random.permutation(train_size)
            x_train = x_train[perm]
            y_train = y_train[perm]

            batched_x_train = x_train[:(self.batches * self.batch_size)].reshape(self.batches, self.batch_size, -1)
            batched_y_train = y_train[:(self.batches * self.batch_size)].reshape(self.batches, self.batch_size)

            for batch, (image, target) in tqdm(enumerate(zip(batched_x_train, batched_y_train))):

                prediction = self.forward(image)
                oh_target = np.eye(10)[target]

                self.back((softmax(prediction) - oh_target))

        return None


    def eval(self, x_test, y_test):
        self.isTraining = False

        prediction = np.argmax(self.forward(x_test), axis = 1)
        correct = np.sum((prediction == y_test).astype(int))

        return correct


def main():
    training_data = pd.read_csv('/Users/william/Desktop/MNIST/mnist_train.csv', header = None)
    x_train, y_train = training_data.values[:, 1:], training_data.values[:, 0] # [row number, column number]
    x_train_std = x_train.std(axis = 0) + 1e-8
    x_train_mean = x_train.mean(axis = 0)
    x_train = (x_train - x_train_mean) / x_train_std # standardized per feature
                        
    testing_data = pd.read_csv('/Users/william/Desktop/MNIST/mnist_test.csv', header = None)
    x_test, y_test = testing_data.values[:, 1:], testing_data.values[:, 0]
    x_test = (x_test - x_train_mean) / x_train_std 

    digits_MLP = MLP([256, 128], 
                        epochs = 12, 
                        batch_size = 32, 
                        learning_rate = 0.01, 
                        dropout_rate = 0.3)

    digits_MLP.train(x_train, y_train)
    correct = digits_MLP.eval(x_test, y_test)
    print(correct)




if __name__ == "__main__":
    main()