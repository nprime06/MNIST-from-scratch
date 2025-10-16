import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import random
import pickle
import copy
import time
from tqdm import tqdm

############
# v1.0: vectorized (v0 commented)
# Changes: vectorized forward and back passes
#          added dropout
############

##### todo list #####
# vectorize batching
# data aug
# final mini batch

##### low priority #####
# check [i, j] vs [i][j] 
# generalize stride & padding in conv functions
# generalize pooling stride & size in pool function 

##### architecture #####
# image -> Conv(32) + Relu -> Conv(32) + Relu -> Max pool
#       -> Conv(64) + Relu -> Conv(64) + Relu -> Max pool 
#       -> Flat + Dense(128) + Relu (dropout) -> out (softmax)

# training hyperparameters (editable)
epochs = 6
batch_size = 32
learning_rate = 0.005
dropout_rate = 0.5

# other stuff (fixed)
train_size = 60000
test_size = 10000

def one_hot_encode(y):
    _ = np.zeros((10,1))
    _[int(y)] = 1
    return _

def relu(x):
    return np.maximum(x,0)

def H(x):
    return (x > 0).astype(int) 

def softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x) + 1e-8)

def conv(in_channel, kernel, bias): # vectorized
    # stride: 1
    # padding: 1 (zeros)
    # kernel size 3 x 3 (preserves in_channel 2d shape)

    C_in, h, w = in_channel.shape
    C_out, _, kh, kw = kernel.shape
    # bias.shape: (C_out, 1)

    padded_in_channel = np.pad(in_channel, ((0, 0), (1, 1), (1, 1)), mode = 'constant')
    windowed_in_channel = sliding_window_view(padded_in_channel, window_shape = (kh, kw), axis = (1, 2)).transpose(1, 2, 0, 3, 4).reshape(h * w, C_in * kh * kw)
    flat_kernel = kernel.reshape(C_out, C_in * kh * kw) # note: array vals still copied by reference (OK here not edited)
    out_channel = (flat_kernel @ windowed_in_channel.T).reshape(C_out, h, w) + bias.reshape(C_out, 1, 1)

    return out_channel
    # out_channel.shape: (C_out, h, w)

'''
def conv_single_channel(in_channel, kernel):
    # stride: 1
    # padding: 1 (zeros)
    # kernel size 3 x 3b

    h, w = in_channel.shape
    kh, kw = kernel.shape 
    k_size = kh * kw


    in_channel = np.pad(in_channel, pad_width = 1, mode = 'constant')
    out_channel = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            out_channel[i, j] = np.sum(in_channel[(i):(i + kh), (j):(j + kw)] * kernel)
    # clean up with matmul (vectorize)
    return out_channel

    # input one channel, output one channel
'''

def back_conv_channel(grad_out_channel, kernel): # vectorized
    C_out, h, w = grad_out_channel.shape
    _, C_in, kh, kw = kernel.shape

    padded_grad_out_channel = np.pad(grad_out_channel, ((0, 0), (1, 1), (1, 1)), mode = 'constant')
    windowed_grad_out_channel = sliding_window_view(padded_grad_out_channel, window_shape = (kh, kw), axis = (1, 2)).transpose(1, 2, 0, 3, 4).reshape(h * w, C_out * kh * kw)
    flipped_kernel = kernel[:, :, ::-1, ::-1]
    flat_flipped_kernel = flipped_kernel.transpose(1, 0, 2, 3).reshape(C_in, C_out * kh * kw) # note: array vals still copied by reference (OK here not edited)
    grad_in_channel = (flat_flipped_kernel @ windowed_grad_out_channel.T).reshape(C_in, h, w)

    return grad_in_channel
    # grad_in_channel.shape: (C_in, h, w)

'''
def back_conv_single_channel(grad_out_channel, kernel):
    n, h, w = grad_out_channel.shape
    _, kh, kw = kernel.shape 

    grad_out_channel = np.pad(grad_out_channel, ((0, 0), (1, 1), (1, 1)), mode = 'constant')
    flipped_kernel = kernel[:, ::-1, ::-1] # note: array vals still copied by reference (OK here not edited)
    grad_in_channel = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            for l in range(n):
                grad_in_channel[i, j] += np.sum(grad_out_channel[l, (i):(i + kh), (j):(j + kw)] * flipped_kernel[l])
            
    # clean up with matmul (vectorize)
    return grad_in_channel

    # input upstream layer of channel gradients, output one channel gradient
'''

def back_conv_kernel(grad_out_channel, in_channel): # vectorized
    C_out, h, w = grad_out_channel.shape
    C_in, _0, _1 = in_channel.shape 
    kh, kw = 3, 3

    padded_in_channel = np.pad(in_channel, ((0, 0), (1, 1), (1, 1)), mode = 'constant')
    windowed_in_channel = sliding_window_view(padded_in_channel, window_shape = (h, w), axis = (1, 2)).reshape(C_in * kh * kw, h * w)
    flat_grad_out_channel = grad_out_channel.reshape(C_out, h * w) # note: array vals still copied by reference (OK here not edited)
    grad_kernel = (flat_grad_out_channel @ windowed_in_channel.T).reshape(C_out, C_in, kh, kw)

    return grad_kernel
    # grad_kernel.shape: (C_out, C_in, kh, kw)

'''
def back_conv_single_kernel(grad_out_channel, in_channel):
    C_out, h, w = grad_out_channel.shape

    in_channel = np.pad(in_channel, pad_width = 1, mode = 'constant')
    kernel = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            kernel[i, j] = np.sum(in_channel[i:(i + h), j:(j + w)] * grad_out_channel)

    return kernel

    # input one upstream channel gradient and lower channel activation, output one kernel gradient
'''

def max_pool(in_channel): # vectorized
    # pool: (2, 2)
    # stride: 2

    C_in, h, w = in_channel.shape
    oh, ow = h // 2, w // 2
    blocked_in_channel = in_channel.reshape(C_in, oh, 2, ow, 2).transpose(0, 1, 3, 2, 4) # note: array vals still copied by reference (OK here not edited)
    out_channel = np.max(blocked_in_channel, axis = (3, 4))
    mask = (blocked_in_channel == out_channel.reshape(C_in, oh, ow, 1, 1)).astype(int).transpose(0, 1, 3, 2, 4).reshape(C_in, h, w)

    return out_channel, mask

'''
def max_pool_single_channel(in_channel): 
    # 2 x 2 pool
    h, w = in_channel.shape # make sure h, w even
    oh, ow = h // 2, w // 2
    out_channel = np.zeros((oh, ow))
    mask = np.zeros((h, w))
    for i in range(oh):
        for j in range(ow):
            out_channel[i, j] = np.max(in_channel[(2 * i):(2 * i + 2), (2 * j):(2 * j + 2)])
            mask[(2 * i):(2 * i + 2), (2 * j):(2 * j + 2)] = (in_channel[(2 * i):(2 * i + 2), (2 * j):(2 * j + 2)] == out_channel[i,j]).astype(int)

    return out_channel, mask
    # clean up mask
    # clean up with matmul (vectorize)
'''


def forward(w, b, image, isTraining):
    # image.shape: (784)

    x = [None] * 15
    pool_mask_cache = [None] * 2
    dropout_mask = None

    if(isTraining):
        dropout_mask = (np.random.randn(128, 1) > dropout_rate).astype(int) / (1 - dropout_rate)
    else:
        dropout_mask = np.ones((128, 1))

    x[0] = image.reshape(1, 28, 28) # image 
    x[1] = conv(x[0], w[0], b[0])
    # x[1] = np.array([np.sum(np.array([conv_single_channel(x[0][j], w[0][i][j]) for j in range(1)]), axis = 0) + b[0][i] for i in range(32)]) # conv_1_2
    x[2] = relu(x[1])
    x[3] = conv(x[2], w[1], b[1])
    #x[3] = np.array([np.sum(np.array([conv_single_channel(x[2][j], w[1][i][j]) for j in range(32)]), axis = 0) + b[1][i] for i in range(32)]) # conv_1_2
    x[4] = relu(x[3])
    x[5], pool_mask_cache[0] = max_pool(x[4])
    # x[5] = np.array([max_pool_single_channel(x[4][i])[0] for i in range(32)]) # max_pool_1
    x[6] = conv(x[5], w[2], b[2])
    # x[6] = np.array([np.sum(np.array([conv_single_channel(x[5][j], w[2][i][j]) for j in range(32)]), axis = 0) + b[2][i] for i in range(64)]) #conv_2_1
    x[7] = relu(x[6])
    x[8] = conv(x[7], w[3], b[3])
    # x[8] = np.array([np.sum(np.array([conv_single_channel(x[7][j], w[3][i][j]) for j in range(64)]), axis = 0) + b[3][i] for i in range(64)]) #conv_2_2
    x[9] = relu(x[8])
    x[10], pool_mask_cache[1] = max_pool(x[9])
    # x[10] = np.array([max_pool_single_channel(x[9][i])[0] for i in range(64)]) # max_pool_2
    x[11] = x[10].reshape(-1, 1)
    x[12] = w[4] @ x[11] + b[4]
    x[13] = relu(x[12]) * dropout_mask
    x[14] = w[5] @ x[13] + b[5] # last layer not activated

    # for i in range(15):
        # print(f'x{i} shape: ', x[i].shape)

    '''
    mask_cache = [None] * 2
    mask_cache[0] = np.array([max_pool_single_channel(x[4][i])[1] for i in range(32)])
    mask_cache[1] = np.array([max_pool_single_channel(x[9][i])[1] for i in range(64)])
    '''

    # for i in range(2):
        # print(f'mask{i} shape: ', mask_cache[i].shape)
    
    return x, pool_mask_cache, dropout_mask


def gradient_calculation(w, b, x, y, pool_mask_cache, dropout_mask):
    grad_x, grad_w, grad_b = copy.deepcopy(x), copy.deepcopy(w), copy.deepcopy(b) # to get same size 

    grad_x[14] = softmax(x[14]) - y
    grad_x[13] = (w[5].T @ grad_x[14]) * dropout_mask
    grad_x[12] = H(x[12]) * grad_x[13]
    grad_x[11] = w[4].T @ grad_x[12]
    grad_x[10] = grad_x[11].reshape(64,7,7)
    grad_x[9] = (pool_mask_cache[1].reshape(64,7,2,7,2).transpose(0,1,3,2,4) * grad_x[10].reshape(64,7,7,1,1)).transpose(0,1,3,2,4).reshape(64,14,14) 
    grad_x[8] = H(x[8]) * grad_x[9]
    grad_x[7] = back_conv_channel(grad_x[8], w[3])
    # grad_x[7] = np.array([back_conv_single_channel(grad_x[8], w[3][:, i]) for i in range(64)])
    grad_x[6] = H(x[6]) * grad_x[7]
    grad_x[5] = back_conv_channel(grad_x[6], w[2])
    # grad_x[5] = np.array([back_conv_single_channel(grad_x[6], w[2][:, i]) for i in range(32)])
    grad_x[4] = (pool_mask_cache[0].reshape(32,14,2,14,2).transpose(0,1,3,2,4) * grad_x[5].reshape(32,14,14,1,1)).transpose(0,1,3,2,4).reshape(32,28,28)
    grad_x[3] = H(x[3]) * grad_x[4]
    grad_x[2] = back_conv_channel(grad_x[3], w[1])
    # grad_x[2] = np.array([back_conv_single_channel(grad_x[3], w[1][:, i]) for i in range(32)])
    grad_x[1] = H(x[1]) * grad_x[2]
    grad_x[0] = None # dont need this
    # for max_pool backprop, alternatively use tiling method (easier, worse)
    # bothers me that max_pool backprop isn't it's own separate function... but it fits in a single line...

    grad_b[5] = grad_x[14]
    grad_b[4] = grad_x[12]
    grad_b[3] = np.sum(grad_x[8], axis = (1, 2)).reshape(-1, 1)
    grad_b[2] = np.sum(grad_x[6], axis = (1, 2)).reshape(-1, 1)
    grad_b[1] = np.sum(grad_x[3], axis = (1, 2)).reshape(-1, 1)
    grad_b[0] = np.sum(grad_x[1], axis = (1, 2)).reshape(-1, 1) # size (32, 1)

    grad_w[5] = grad_x[14] @ x[13].T
    grad_w[4] = grad_x[12] @ x[11].T
    grad_w[3] = back_conv_kernel(grad_x[8], x[7])
    grad_w[2] = back_conv_kernel(grad_x[6], x[5])
    grad_w[1] = back_conv_kernel(grad_x[3], x[2])
    grad_w[0] = back_conv_kernel(grad_x[1], x[0])

    '''
    for i in range(64):
        for j in range(64):
            grad_w[3][i, j] = back_conv_single_kernel(grad_x[8][i], x[7][j])

    for i in range(64):
        for j in range(32):
            grad_w[2][i, j] = back_conv_single_kernel(grad_x[6][i], x[5][j])

    for i in range(32):
        for j in range(32):
            grad_w[1][i, j] = back_conv_single_kernel(grad_x[3][i], x[2][j])

    for i in range(32):
        for j in range(1):
            grad_w[0][i, j] = back_conv_single_kernel(grad_x[1][i], x[0][j])
    
    # clean up with matmul (vectorize)
    '''
    return grad_w, grad_b


def train(x_train, y_train):
    w = [np.sqrt(2. / (1 * 3 * 3)) * np.random.randn(32, 1, 3, 3), np.sqrt(2. / (32 * 3 * 3)) * np.random.randn(32, 32, 3, 3), np.sqrt(2. / (32 * 3 * 3)) * np.random.randn(64, 32, 3, 3), np.sqrt(2. / (64 * 3 * 3)) * np.random.randn(64, 64, 3, 3), np.sqrt(2. / (3136)) * np.random.randn(128, 3136), np.sqrt(2. / (128)) * np.random.randn(10, 128)]
    b = [np.zeros((32, 1)), np.zeros((32, 1)), np.zeros((64, 1)), np.zeros((64, 1)), np.zeros((128, 1)), np.zeros((10, 1))]
    # initialize weights

    for epoch in range(epochs):
        perm = np.random.permutation(train_size)
        x_train = x_train[perm]
        y_train = y_train[perm]

        grad_w_batch = [np.zeros((32, 1, 3, 3)), np.zeros((32, 32, 3, 3)), np.zeros((64, 32, 3, 3)), np.zeros((64, 64, 3, 3)), np.zeros((128, 3136)), np.zeros((10, 128))]
        grad_b_batch = [np.zeros((32, 1)), np.zeros((32, 1)), np.zeros((64, 1)), np.zeros((64, 1)), np.zeros((128, 1)), np.zeros((10, 1))]
        
        for i, (image, target) in tqdm(enumerate(zip(x_train, y_train))): 

            x, pool_mask_cache, dropout_mask = forward(w, b, image, True)

            # for j in range(15):
                # print(f'x{j} shape: ', x[j].shape)

            y = one_hot_encode(target)
            grad_w, grad_b = gradient_calculation(w, b, x, y, pool_mask_cache, dropout_mask)

            if i % (train_size * epochs // 10000) == 0:
                progress = (epoch / epochs) + (1 / epochs) * (i / train_size)
                # print(target, softmax(x[-1]).argmax(), f"{round(progress * 100, 2)}%")

            # for j in range(6):
                # print(f'grad_w_batch{j}: ', grad_w_batch[j].shape)
                # print(f'grad_b_batch{j}: ', grad_b_batch[j].shape)
                # print(f'grad_w{j}: ', grad_w[j].shape)
                # print(f'grad_b{j}: ', grad_b[j].shape)

            for j in range(6):
                grad_w_batch[j] += grad_w[j]
                grad_b_batch[j] += grad_b[j]
            
            if ((i + 1) % batch_size == 0):
                for j in range(6): 
                    w[j] -= learning_rate * (grad_w_batch[j] / batch_size)
                    b[j] -= learning_rate * (grad_b_batch[j] / batch_size)

                grad_w_batch = [np.zeros((32, 1, 3, 3)), np.zeros((32, 32, 3, 3)), np.zeros((64, 32, 3, 3)), np.zeros((64, 64, 3, 3)), np.zeros((128, 3136)), np.zeros((10, 128))]
                grad_b_batch = [np.zeros((32, 1)), np.zeros((32, 1)), np.zeros((64, 1)), np.zeros((64, 1)), np.zeros((128, 1)), np.zeros((10, 1))]

        # process remaining data in last "mini batch"
    return w, b


def test():
    training_data = pd.read_csv("mnist_train.csv", header = None)
    x_train, y_train = training_data.values[:, 1:] / 255, training_data.values[:, 0] # [row number, column number]
    # x_train_std = x_train.std(axis = 0) + 1e-8
    # x_train_mean = x_train.mean(axis = 0)
    # x_train = (x_train - x_train_mean) / x_train_std
                        
    testing_data = pd.read_csv("mnist_test.csv", header = None)
    x_test, y_test = testing_data.values[:, 1:] / 255, testing_data.values[:, 0]
    # x_test = (x_test - x_train_mean) / x_train_std # standardize 
    
    w, b = train(x_train, y_train)

    wrong = 0
    for i, (image, target) in enumerate(zip(x_test, y_test)):
        x, _0, _1 = forward(w, b, image, False)
        if(softmax(x[-1]).argmax() != target):
            wrong += 1

    print("epochs: " + str(epochs))
    print("accuracy: " + str(wrong))

    with open("MNIST_CNN_wb.pkl", "wb") as f:
        pickle.dump({"weights": w, "biases": b}, f)

test()

