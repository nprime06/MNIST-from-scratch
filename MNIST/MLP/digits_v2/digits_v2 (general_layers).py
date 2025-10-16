import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# note: functional implementation

# Not yet implemented:
# dropout
# batch vectorize
# momentum, decaying learning rate
# data aug

def one_hot_encode(y):
    _ = np.zeros((10,1))
    _[int(y)] = 1
    return _

def relu(x):
    return np.maximum(x,0)

def Id(x):
    return x

def H(x):
    return (x > 0).astype(int)

def one(x):
    return np.ones(x.shape)

def softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x) + 1e-8)

# parameters (editable)
neurons = [784, 128, 10]
epochs = 6
batch_size = 32
learning_rate = 0.005

# other stuff (fixed)
train_size = 60000
test_size = 10000
layers = len(neurons)
activations = [relu] * (layers - 2) + [Id]
dactivations = [H] * (layers - 2) + [one]


def forward(w, b, x):
    for i in range(layers - 1):
        x[i+1] = activations[i](w[i] @ x[i] + b[i])
    return x


def gradient_calculation(w, b, x, y):
    grad_x, grad_w, grad_b = [None] * layers, [None] * (layers - 1), [None] * (layers - 1)
    
    grad_x[-1] = softmax(x[-1]) - y
    for i in range(layers - 2, -1, -1):
        grad_x[i] = w[i].T @ (dactivations[i](x[i + 1]) * grad_x[i + 1])

    for i in range(layers - 1):
        grad_w[i] = (dactivations[i](x[i + 1]) * grad_x[i + 1]) @ x[i].T
        grad_b[i] = (dactivations[i](x[i + 1]) * grad_x[i + 1])
    
    return grad_w, grad_b


def train(x_train, y_train):
    w = [np.sqrt(2. / neurons[i]) * np.random.randn(neurons[i + 1], neurons[i]) for i in range(layers - 1)]
    b = [np.zeros((neurons[i], 1)) for i in range(1, layers, 1)]
    # b = [0.01 * np.random.randn(neurons[i], 1) for i in range(1, layers, 1)]

    for epoch in tqdm(range(epochs)):
        perm = np.random.permutation(train_size)
        x_train = x_train[perm]
        y_train = y_train[perm]

        grad_w_batch = [np.zeros((neurons[i + 1], neurons[i])) for i in range(layers - 1)]
        grad_b_batch = [np.zeros((neurons[i], 1)) for i in range(1, layers, 1)]
        
        for i, (image, target) in tqdm(enumerate(zip(x_train, y_train))):
            x = forward(w, b, [image.reshape(neurons[0], 1)] + [None] * (layers - 1))
            y = one_hot_encode(target)
            grad_w, grad_b = gradient_calculation(w, b, x, y)


            # if i % 500 == 0:
                # progress = (epoch / epochs) + (1 / epochs) * (i / train_size)
                # print(target, softmax(x[-1]).argmax(), f"{round(progress * 100, 2)}%")

            for j in range(layers - 1):
                grad_w_batch[j] += grad_w[j]
                grad_b_batch[j] += grad_b[j]
            
            if ((i + 1) % batch_size == 0):
                for j in range(layers - 1):
                    w[j] -= learning_rate * (grad_w_batch[j] / batch_size)
                    b[j] -= learning_rate * (grad_b_batch[j] / batch_size)

                grad_w_batch = [np.zeros((neurons[i + 1], neurons[i])) for i in range(layers - 1)]
                grad_b_batch = [np.zeros((neurons[i], 1)) for i in range(1, layers, 1)]

        # process remaining data in last "mini batch"

    return w, b


def test():
    training_data = pd.read_csv('../../mnist_train.csv', header = None)
    x_train, y_train = training_data.values[:, 1:], training_data.values[:, 0] # [row number, column number]
    x_train_std = x_train.std(axis = 0) + 1e-8
    x_train_mean = x_train.mean(axis = 0)
    x_train = (x_train - x_train_mean) / x_train_std # standardized per feature
                      
    testing_data = pd.read_csv('../../mnist_test.csv', header = None)
    x_test, y_test = testing_data.values[:, 1:], testing_data.values[:, 0]
    x_test = (x_test - x_train_mean) / x_train_std # standardize
    
    w, b = train(x_train, y_train)

    wrong = 0
    for i, (image, target) in enumerate(zip(x_test, y_test)):


        x = [None] * layers
        x[0] = image.reshape(neurons[0], 1)
        x = forward(w, b, x)
        if(softmax(x[-1]).argmax() != target):
            wrong += 1

    print("epochs: " + str(epochs))
    print("accuracy: " + str(wrong))

    with open(f"{layers - 2}layerNN.pkl", "wb") as f:
        pickle.dump({"neurons": neurons, "weights": w, "biases": b}, f)

test()


