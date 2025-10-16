import numpy as np
import pandas as pd
import random
from tqdm import tqdm
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def one_hot_encode(y):
    _ = np.zeros((10, 1))
    _[int(y)] = 1
    return _

def forward(w, b, x):
    y_ = sigmoid(w @ x + b)
    return w, b, x, y_

def gradient_calculation(y, y_, x):
    grad_w = (y_ - y) @ x.T
    grad_b = (y_ - y)
    return grad_w, grad_b

def parameter_update(w, grad_w, b, grad_b, learning_rate):
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    return w, b

def log_loss(y, y_):
    y, y_ = y.reshape(10), y_.reshape(10)
    
    loss = -1 * ((y @ np.log(y_)) + ((1 - y) @ np.log(1 - y_)))
    return loss

def train(x_train, y_train, epochs, learning_rate):
    w = np.random.randn(10, 784)
    b = np.random.randn(10, 1)

    for epoch in tqdm(range(epochs)):
        for i, (image, target) in enumerate(zip(x_train, y_train)):
            image = image.reshape(784, 1)
            w, b, x, y_ = forward(w, b, image)
            y = one_hot_encode(target)
            # if i % 500 == 0:
                # print(target, y_.argmax())

            grad_w, grad_b = gradient_calculation(y, y_, x)
            w, b = parameter_update(w, grad_w, b, grad_b, learning_rate)

    return w, b
            
            
def test():
    epochs = 6
    learning_rate = 0.005

    training_data = pd.read_csv("/Users/william/Desktop/MNIST/mnist_train.csv", header = None)
    x_train, y_train = training_data.values[:, 1:], training_data.values[:, 0]
    x_train_std = np.std(x_train)
    x_train_mean = np.mean(x_train)
    x_train = (x_train - x_train_mean) / x_train_std
                        
    testing_data = pd.read_csv("/Users/william/Desktop/MNIST/mnist_test.csv", header = None)
    x_test, y_test = testing_data.values[:, 1:], testing_data.values[:, 0]
    x_test_std = np.std(x_test)
    x_test_mean = np.mean(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    
    w, b = train(x_train, y_train, epochs, learning_rate)
    
    wrong = 0
    for i, (image, target) in enumerate(zip(x_test, y_test)):
        image = image.reshape(784, 1)
        y_ = forward(w, b, image)[-1]
        if(y_.argmax() != target):
            wrong += 1

    print("accuracy: ")
    print(wrong)
            
test()
