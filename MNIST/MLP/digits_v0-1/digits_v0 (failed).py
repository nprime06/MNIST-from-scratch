import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

training_data = pd.read_csv("mnist_train.csv", header = None)
training_examples = 60000

testing_data = pd.read_csv("mnist_test.csv", header = None)
testing_examples = 10000

epochs = 10
batch_size = 60
learning_rate = 0.001


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f(x):
    return x * (1-x)

a0 = np.zeros((784))
a1 = np.zeros((16))
a2 = np.zeros((16))
a3 = np.zeros((10)) # activations
y = np.zeros((10)) # target

b1 = np.zeros((16))
b2 = np.zeros((16))
b3 = np.zeros((10)) # biases

e0 = np.zeros((784, 16))
e1 = np.zeros((16, 16))
e2 = np.zeros((16, 10)) # weights

for i in range(784):
    for j in range(16):
        e0[i, j] = np.random.normal() / 800

for i in range(16):
    for j in range(16):
        e1[i, j] = np.random.normal() / 32

for i in range(16):
    for j in range(10):
        e2[i, j] = np.random.normal() / 26

dcda1 = np.zeros((16))
dcda2 = np.zeros((16))
dcda3 = np.zeros((10)) # dC/da

dcdb1 = np.zeros((16))
dcdb2 = np.zeros((16))
dcdb3 = np.zeros((10))

dcde0 = np.zeros((784, 16))
dcde1 = np.zeros((16, 16))
dcde2 = np.zeros((16, 10))
# accumulated gradients (multiply by lr and divide by batch size later)



def read_data(index):
    global a0, y

    a0 = np.array(training_data.iloc[index, 1:])/255
    y[:] = 0
    y[training_data.iloc[index, 0]] = 1



def learn(): # accumulates gradients from one training data loaded by read_data
    global a1, a2, a3, b1, b2, b3, e0, e1, e2, dcda1, dcda2, dcda3, dcdb1, dcdb2, dcdb3, dcde0, dcde1, dcde2
    
    a1 = sigmoid(e0.T @ a0 + b1)
    a2 = sigmoid(e1.T @ a1 + b2)
    a3 = sigmoid(e2.T @ a2 + b3)

    dcda3 = 2 * (a3 - y)
    dcda2 = e2 @ (f(a3) * dcda3)
    dcda1 = e1 @ (f(a2) * dcda2)

    dcdb1 += dcda1 * f(a1)
    dcdb2 += dcda2 * f(a2)
    dcdb3 += dcda3 * f(a3)

    dcde0 += np.outer(a0, dcda1 * f(a1))
    dcde1 += np.outer(a1, dcda2 * f(a2))
    dcde2 += np.outer(a2, dcda3 * f(a3))
    print ((a3-y) @ (a3-y))
    return ((a3 - y) @ (a3 - y))



def train():
    global b1, b2, b3, e0, e1, e2, dcdb1, dcdb2, dcdb3, dcde0, dcde1, dcde2
    
    order = list(range(training_examples))

    avg_cost = 1000
    curr_batch = 1
    x = np.array([0])
    y = np.array([10])
    
    for epoch in range(epochs):
        random.shuffle(order)

        for batch in range(int(training_examples / batch_size)):
            
            dcdb1[:] = 0
            dcdb2[:] = 0
            dcdb3[:] = 0
            dcde0[:] = 0
            dcde1[:] = 0
            dcde2[:] = 0

            for batch_index in range(batch_size):

                read_data(batch * batch_size + batch_index)
                avg_cost += learn()
            
            b1 -= (learning_rate / batch_size) * dcdb1
            b2 -= (learning_rate / batch_size) * dcdb2
            b3 -= (learning_rate / batch_size) * dcdb3
            e0 -= (learning_rate / batch_size) * dcde0
            e1 -= (learning_rate / batch_size) * dcde1
            e2 -= (learning_rate / batch_size) * dcde2


            if(curr_batch % 100 == 0):
                x = np.append(x, curr_batch / 100)
                y = np.append(y, avg_cost / (100 * batch_size))
                avg_cost = 0

                #print(curr_batch / (epochs * int(training_examples / batch_size)))
            
            curr_batch += 1

    fig, ax = plt.subplots()

    print(y)
    line, = ax.plot(x, y)
    plt.show()
    


def test(test_cases):    
    order = random.sample(list(range(testing_examples)), test_cases)

    for test_case in order:

        a0 = np.array(testing_data.iloc[test_case, 1:])/255
        real = testing_data.iloc[test_case, 0]

        a1 = sigmoid(e0.T @ a0 + b1)
        a2 = sigmoid(e1.T @ a1 + b2)
        a3 = sigmoid(e2.T @ a2 + b3)

        print(a3)
        print(real)
    
    # load up the test file
    # shuffle indices, choose # of test cases we want
    # read data a0
    # feed forward
    # display a3? and predicted result against real 
    # possibly display the data in handwriting form

train()
test(5)

    
