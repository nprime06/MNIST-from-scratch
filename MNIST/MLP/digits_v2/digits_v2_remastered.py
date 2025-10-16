import numpy as np
import pandas as pd
from tqdm import tqdm

# note: functional implementation 
# no batch norm, dropout, adam
# just bare bones

def softmax(x, axis = 1, eps = 1e-8, dtype = np.float32):
    x_max = np.max(x, axis = axis, keepdims = True)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x, axis = axis, dtype = dtype, keepdims = True) + dtype(eps))

def relu(x):
    return np.maximum(x, 0)

def H(x): 
    return (x > 0).astype(np.float32)

def sigmoid(x):
    return (1 / (1 + np.exp(-1 * x)))

def dsigmoid(x):
    return x * (1 - x)

def Id(x):
    return x

def one(x):
    return np.ones_like(x)


# model hyperparams
neurons = [784, 128, 10]
layers = len(neurons)
activations = [relu] * (layers - 2) + [Id]
dactivations = [H] * (layers - 2) + [one]

def forward(w, b, x):
    # change x in place
    for i in range(layers - 1):
        x[i + 1].fill(0)
        x[i + 1] += activations[i](x[i] @ w[i].T + b[i])
    return x[-1]

def grad_calc(w, b, x, y, learning_rate):
    curr_grad = softmax(x[-1]) - y
    for i in reversed(range(layers - 1)):
        next_grad = (dactivations[i](x[i + 1]) * curr_grad) @ w[i]
        w[i] -= learning_rate * (curr_grad.T @ x[i]) / curr_grad.shape[0]
        b[i] -= learning_rate * np.mean(curr_grad, axis = 0, keepdims = True, dtype = np.float32)
        curr_grad = next_grad

def main():
    train_size = 60000
    test_size = 10000
    dtype = np.float32
    eps = dtype(1e-8)

    training_data = pd.read_csv('../../mnist_train.csv', header = None)
    x_train, y_train = training_data.values[:, 1:].astype(dtype), training_data.values[:, 0].astype(np.int32) # [row number, column number]
    x_train_var = x_train.var(axis = 0, dtype = dtype)
    x_train_mean = x_train.mean(axis = 0, dtype = dtype)
    x_train = (x_train - x_train_mean) / np.sqrt(x_train_var + eps) # standardized per feature
    y_train = np.eye(10, dtype = dtype)[y_train] # oh encode

    testing_data = pd.read_csv('../../mnist_test.csv', header = None)
    image_test, y_test = testing_data.values[:, 1:].astype(dtype), testing_data.values[:, 0].astype(np.int32)
    image_test = (image_test - x_train_mean) / np.sqrt(x_train_var + eps)


    x = [None] * layers
    x_test = [None] * layers
    w = [None] * (layers - 1)
    b = [None] * (layers - 1)

    # training hyperparams
    epochs = 100
    batch_size = 32
    batches = train_size // batch_size
    learning_rate = 0.01

    # init
    for i in range(layers):
        x[i] = np.zeros((batch_size, neurons[i])).astype(dtype)
        x_test[i] = np.zeros((test_size, neurons[i])).astype(dtype)

    for i in range(layers - 1):
        w[i] = (np.sqrt(2 / neurons[i]) * np.random.randn(neurons[i + 1], neurons[i])).astype(dtype)
        # w[i] = (0.01 * np.random.randn(neurons[i + 1], neurons[i])).astype(dtype)
        b[i] = np.zeros((1, neurons[i + 1])).astype(dtype)

    for epoch in tqdm(range(epochs)):
        perm = np.random.permutation(train_size)
        x_train = x_train[perm]
        y_train = y_train[perm]

        batched_x_train = x_train[:(batches * batch_size)].reshape(batches, batch_size, -1)
        batched_y_train = y_train[:(batches * batch_size)].reshape(batches, batch_size, -1)

        for batch, (image, target) in enumerate(zip(batched_x_train, batched_y_train)):
            x[0].fill(0)
            x[0] += image

            forward(w, b, x)
            grad_calc(w, b, x, target, learning_rate)

        x_test[0].fill(0)
        x_test[0] += image_test
        y_pred = np.argmax(forward(w, b, x_test), axis = 1).astype(int)
        #print(f'Epoch {epoch} test accuracy: {np.sum((y_pred == y_test).astype(int))}')
        print(f'{np.sum((y_pred == y_test).astype(int))}')



if __name__ == '__main__':
    main()