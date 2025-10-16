import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import random
import pickle
import time
import copy
from tqdm import tqdm

############
# v1.1: fully vectorized
############

##### todo list #####
# data aug

##### architecture #####
# image -> Conv(32) + Relu -> Conv(32) + Relu -> Max pool
#       -> Conv(64) + Relu -> Conv(64) + Relu -> Max pool 
#       -> Flat + Dense(128) + Relu (dropout) -> out (softmax)

# hyperparameters in training loop


def relu(x):
    return np.maximum(x,0)

def H(x):
    return (x > 0).astype(int) 

def softmax(x):
    x_max = np.max(x, axis = 1, keepdims = True)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x, axis = 1).reshape(-1, 1) + 1e-8)

def conv(in_channel, kernel, bias): # vectorized
    # stride: 1
    # padding: 1 (zeros)
    # kernel size 3 x 3 (preserves in_channel 2d shape)

    N, C_in, h, w = in_channel.shape
    C_out, _, kh, kw = kernel.shape
    # bias.shape: (C_out,)

    padded_in_channel = np.pad(in_channel, ((0, 0), (0, 0), (1, 1), (1, 1)), mode = 'constant')
    windowed_in_channel = sliding_window_view(padded_in_channel, window_shape = (kh, kw), axis = (2, 3)).transpose(0, 2, 3, 1, 4, 5).reshape(N * h * w, C_in * kh * kw)
    flat_kernel = kernel.reshape(C_out, C_in * kh * kw) # note: array vals still copied by reference (OK here not edited)
    out_channel = (flat_kernel @ windowed_in_channel.T).reshape(C_out, N, h, w).transpose(1, 0, 2, 3) + bias.reshape(1, C_out, 1, 1)

    return out_channel
    # out_channel.shape: (N, C_out, h, w)

def back_conv_channel(grad_out_channel, kernel): # vectorized
    N, C_out, h, w = grad_out_channel.shape
    _, C_in, kh, kw = kernel.shape

    padded_grad_out_channel = np.pad(grad_out_channel, ((0, 0), (0, 0), (1, 1), (1, 1)), mode = 'constant')
    windowed_grad_out_channel = sliding_window_view(padded_grad_out_channel, window_shape = (kh, kw), axis = (2, 3)).transpose(0, 2, 3, 1, 4, 5).reshape(N * h * w, C_out * kh * kw)
    flipped_kernel = kernel[:, :, ::-1, ::-1]
    flat_flipped_kernel = flipped_kernel.transpose(1, 0, 2, 3).reshape(C_in, C_out * kh * kw) # note: array vals still copied by reference (OK here not edited)
    grad_in_channel = (flat_flipped_kernel @ windowed_grad_out_channel.T).reshape(C_in, N, h, w).transpose(1, 0, 2, 3)

    return grad_in_channel
    # grad_in_channel.shape: (N, C_in, h, w)

def back_conv_kernel(grad_out_channel, in_channel): # vectorized
    N, C_out, h, w = grad_out_channel.shape
    _0, C_in, _1, _2 = in_channel.shape 
    kh, kw = 3, 3

    padded_in_channel = np.pad(in_channel, ((0, 0), (0, 0), (1, 1), (1, 1)), mode = 'constant') 
    windowed_in_channel = sliding_window_view(padded_in_channel, window_shape = (h, w), axis = (2, 3)).transpose(1, 2, 3, 0, 4, 5).reshape(C_in * kh * kw, N * h * w)
    flat_grad_out_channel = grad_out_channel.transpose(1, 0, 2, 3).reshape(C_out, N * h * w) # note: array vals still copied by reference (OK here not edited)
    grad_kernel = (flat_grad_out_channel @ windowed_in_channel.T).reshape(C_out, C_in, kh, kw)

    return grad_kernel
    # grad_kernel.shape: (C_out, C_in, kh, kw)

def max_pool(in_channel): # vectorized
    # pool: (2, 2)
    # stride: 2

    N, C_in, h, w = in_channel.shape
    oh, ow = h // 2, w // 2
    blocked_in_channel = in_channel.reshape(N, C_in, oh, 2, ow, 2).transpose(0, 1, 2, 4, 3, 5) # note: array vals still copied by reference (OK here not edited)
    out_channel = np.max(blocked_in_channel, axis = (4, 5))
    mask = (blocked_in_channel == out_channel.reshape(N, C_in, oh, ow, 1, 1)).astype(int).transpose(0, 1, 2, 4, 3, 5).reshape(N, C_in, h, w)

    return out_channel, mask
    # out_channel.shape: (N, C_in, oh, ow)
    # mask.shape: (N, C_in, h, w)

def forward(w, b, image, dropout_rate): 
    N, _ = image.shape

    x = [None] * 15
    pool_mask_cache = [None] * 2
    dropout_mask = None

    dropout_mask = (np.random.randn(N, 128) >= dropout_rate).astype(int) / (1 - dropout_rate)

    x[0] = image.reshape(-1, 1, 28, 28) # image 
    x[1] = conv(x[0], w[0], b[0])
    x[2] = relu(x[1])
    x[3] = conv(x[2], w[1], b[1])
    x[4] = relu(x[3])
    x[5], pool_mask_cache[0] = max_pool(x[4])
    x[6] = conv(x[5], w[2], b[2])
    x[7] = relu(x[6])
    x[8] = conv(x[7], w[3], b[3])
    x[9] = relu(x[8])
    x[10], pool_mask_cache[1] = max_pool(x[9])
    x[11] = x[10].reshape(N, -1)
    x[12] = x[11] @ w[4].T + b[4].reshape(1, -1)
    x[13] = relu(x[12]) * dropout_mask
    x[14] = x[13] @ w[5].T + b[5].reshape(1, -1) # last layer not activated
    
    return x, pool_mask_cache, dropout_mask


def gradient_calculation(w, b, x, y, pool_mask_cache, dropout_mask):
    grad_x, grad_w, grad_b = copy.deepcopy(x), copy.deepcopy(w), copy.deepcopy(b) # to get same size 

    N, _ = x[14].shape

    grad_x[14] = softmax(x[14]) - y
    grad_x[13] = (grad_x[14] @ w[5]) * dropout_mask
    grad_x[12] = H(x[12]) * grad_x[13]
    grad_x[11] = grad_x[12] @ w[4]
    grad_x[10] = grad_x[11].reshape(N, 64, 7, 7)
    grad_x[9] = (pool_mask_cache[1].reshape(N, 64, 7, 2, 7, 2).transpose(0, 1, 2, 4, 3, 5) * grad_x[10].reshape(N, 64, 7, 7, 1, 1)).transpose(0, 1, 2, 4, 3, 5).reshape(N, 64, 14, 14)
    grad_x[8] = H(x[8]) * grad_x[9]
    grad_x[7] = back_conv_channel(grad_x[8], w[3])
    grad_x[6] = H(x[6]) * grad_x[7]
    grad_x[5] = back_conv_channel(grad_x[6], w[2])
    grad_x[4] = (pool_mask_cache[0].reshape(N, 32, 14, 2, 14, 2).transpose(0, 1, 2, 4, 3, 5) * grad_x[5].reshape(N, 32, 14, 14, 1, 1)).transpose(0, 1, 2, 4, 3, 5).reshape(N, 32, 28, 28)
    grad_x[3] = H(x[3]) * grad_x[4]
    grad_x[2] = back_conv_channel(grad_x[3], w[1])
    grad_x[1] = H(x[1]) * grad_x[2]
    grad_x[0] = None # dont need this
    # for max_pool backprop, alternatively use tiling method (easier, worse)

    grad_b[5] = np.sum(grad_x[14], axis = 0)
    grad_b[4] = np.sum(grad_x[12], axis = 0)
    grad_b[3] = np.sum(grad_x[8], axis = (0, 2, 3))
    grad_b[2] = np.sum(grad_x[6], axis = (0, 2, 3))
    grad_b[1] = np.sum(grad_x[3], axis = (0, 2, 3))
    grad_b[0] = np.sum(grad_x[1], axis = (0, 2, 3)) # needs to have size (C, )

    grad_w[5] = grad_x[14].T @ x[13]
    grad_w[4] = grad_x[12].T @ x[11]
    grad_w[3] = back_conv_kernel(grad_x[8], x[7])
    grad_w[2] = back_conv_kernel(grad_x[6], x[5])
    grad_w[1] = back_conv_kernel(grad_x[3], x[2])
    grad_w[0] = back_conv_kernel(grad_x[1], x[0])

    return grad_w, grad_b


def train(x_train, y_train):
    # training hyperparameters (editable)
    epochs = 6
    batch_size = 32
    learning_rate = 0.005
    dropout_rate = 0.5

    # other stuff (fixed)
    train_size = 60000
    batches = train_size // batch_size


    w = [np.sqrt(2. / (1 * 3 * 3)) * np.random.randn(32, 1, 3, 3), np.sqrt(2. / (32 * 3 * 3)) * np.random.randn(32, 32, 3, 3), np.sqrt(2. / (32 * 3 * 3)) * np.random.randn(64, 32, 3, 3), np.sqrt(2. / (64 * 3 * 3)) * np.random.randn(64, 64, 3, 3), np.sqrt(2. / (3136)) * np.random.randn(128, 3136), np.sqrt(2. / (128)) * np.random.randn(10, 128)]
    b = [np.zeros((32,)), np.zeros((32,)), np.zeros((64,)), np.zeros((64,)), np.zeros((128,)), np.zeros((10,))]
    # initialize weights

    for epoch in tqdm(range(epochs)):
        perm = np.random.permutation(train_size)
        x_train = x_train[perm]
        y_train = y_train[perm]

        batched_x_train = x_train[:(batches * batch_size)].reshape(batches, batch_size, -1)
        batched_y_train = y_train[:(batches * batch_size)].reshape(batches, batch_size)
        
        for batch, (image, target) in tqdm(enumerate(zip(batched_x_train, batched_y_train))): 

            x, pool_mask_cache, dropout_mask = forward(w, b, image, dropout_rate)

            y = np.eye(10)[target] # one hot encode
            grad_w, grad_b = gradient_calculation(w, b, x, y, pool_mask_cache, dropout_mask) # batched

            for j in range(6): 
                w[j] -= learning_rate * (grad_w[j] / batch_size)
                b[j] -= learning_rate * (grad_b[j] / batch_size)

            if batch % (batches * epochs // 10000) == 0:
                progress = (epoch / epochs) + (1 / epochs) * (batch / batches)
                # print(target[-1], softmax(x[-1][-1].reshape(1, 10)).argmax(), f"{round(progress * 100, 2)}%")

    return w, b


def test():
    training_data = pd.read_csv('/Users/william/Desktop/MNIST/mnist_train.csv', header = None)
    x_train, y_train = training_data.values[:, 1:], training_data.values[:, 0] # [row number, column number]
    x_train_std = x_train.std(axis = 0) + 1e-8
    x_train_mean = x_train.mean(axis = 0)
    x_train = (x_train - x_train_mean) / x_train_std # standardized per feature
                        
    testing_data = pd.read_csv('/Users/william/Desktop/MNIST/mnist_test.csv', header = None)
    x_test, y_test = testing_data.values[:, 1:], testing_data.values[:, 0]
    x_test = (x_test - x_train_mean) / x_train_std 
    test_size = 10000
    
    w, b = train(x_train, y_train)

    
    # evaluation
    x_test_prop, _0, _1 = forward(w, b, x_test, 0)
    test_predict = np.argmax(x_test_prop[-1], axis = 1)
    correct = np.sum((test_predict == y_test).astype(int))
    print(f"Test set accuracy: {round((correct / test_size) * 100.0, 2)}% ({test_size - correct} wrong)")

    with open("MNIST_CNN_wb.pkl", "wb") as f:
        pickle.dump({"weights": w, "biases": b}, f)

test()

