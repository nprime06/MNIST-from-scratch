import numpy as np
import pandas as pd
import random
import pickle
import copy
from tqdm import tqdm

############
# v0: not optimized to use vector operations
############

#todo list
# vectorize loops
# some reshapes maybe questionable
# check [i, j] vs [i][j] (low priority)
# generalize stride & padding in conv functions
# generalize pooling stride & size in pool function 



def one_hot_encode(y):
    _ = np.zeros((10,1))
    _[int(y)] = 1
    return _

def relu(x):
    return np.maximum(x,0)

def Id(x):
    return x

def H(x):
    return (x > 0).astype(int) # 1 x > 0

def one(x):
    return np.ones(x.shape)

def softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x) + 1e-8)

# parameters (editable)
# neurons = [784, 800, 10]
epochs = 12
batch_size = 32
learning_rate = 0.005
dropout_rate = 0.5

# other stuff (fixed)
train_size = 60000
test_size = 10000
# layers = len(neurons)
# activations = [relu] * (layers - 2) + [Id]
# dactivations = [H] * (layers - 2) + [one]




def conv_single_channel(in_channel, kernel):
    # stride: 1
    # padding: 1 (zeros)

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

def back_conv_single_channel(grad_out_channel, kernel):
    n, h, w = grad_out_channel.shape
    _, kh, kw = kernel.shape 


    grad_out_channel = np.pad(grad_out_channel, ((0, 0), (1, 1), (1, 1)), mode = 'constant')
    flipped_kernel = kernel[:, ::-1, ::-1] # note: inner stuff still copied by reference (irrelevant here as we don't edit flipped_kernel)
    grad_in_channel = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            for l in range(n):
                grad_in_channel[i, j] += np.sum(grad_out_channel[l, (i):(i + kh), (j):(j + kw)] * flipped_kernel[l])
            
    # clean up with matmul (vectorize)
    return grad_in_channel

    # input upstream layer of channel gradients, output one channel gradient



def back_conv_single_kernel(grad_out_channel, in_channel):
    h, w = grad_out_channel.shape

    in_channel = np.pad(in_channel, pad_width = 1, mode = 'constant')
    kernel = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            kernel[i, j] = np.sum(in_channel[i:(i + h), j:(j + w)] * grad_out_channel)

    return kernel

    # input one upstream channel gradient and lower channel activation, output one kernel gradient



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


def forward(w, b, image):
    # image: (784)
    
    x = [None] * 15
    # last layer not activated

    x[0] = image.reshape(28, 28) # image 
    x[1] = np.array([conv_single_channel(x[0], w[0][i]) + b[0][i] for i in range(32)]) # conv_1_1
    x[2] = relu(x[1])
    x[3] = np.array([np.sum(np.array([conv_single_channel(x[2][j], w[1][i][j]) for j in range(32)]), axis = 0) + b[1][i] for i in range(32)]) # conv_1_2
    x[4] = relu(x[3])
    x[5] = np.array([max_pool_single_channel(x[4][i])[0] for i in range(32)]) # max_pool_1
    x[6] = np.array([np.sum(np.array([conv_single_channel(x[5][j], w[2][i][j]) for j in range(32)]), axis = 0) + b[2][i] for i in range(64)]) #conv_2_1
    x[7] = relu(x[6])
    x[8] = np.array([np.sum(np.array([conv_single_channel(x[7][j], w[3][i][j]) for j in range(64)]), axis = 0) + b[3][i] for i in range(64)]) #conv_2_2
    x[9] = relu(x[8])
    x[10] = np.array([max_pool_single_channel(x[9][i])[0] for i in range(64)]) # max_pool_2
    x[11] = x[10].reshape(-1, 1)
    x[12] = w[4] @ x[11] + b[4]
    x[13] = relu(x[12])
    x[14] = w[5] @ x[13] + b[5]
    # size of b could be an issue for x1, x3, x6, x8

    # for i in range(15):
        # print(f'x{i} shape: ', x[i].shape)

    mask_cache = [None] * 2
    mask_cache[0] = np.array([max_pool_single_channel(x[4][i])[1] for i in range(32)])
    mask_cache[1] = np.array([max_pool_single_channel(x[9][i])[1] for i in range(64)])

    # for i in range(2):
        # print(f'mask{i} shape: ', mask_cache[i].shape)
    
    return x, mask_cache



def gradient_calculation(w, b, x, y, mask_cache):
    grad_x, grad_w, grad_b = copy.deepcopy(x), copy.deepcopy(w), copy.deepcopy(b) # to get same size 

    grad_x[14] = softmax(x[14]) - y
    grad_x[13] = w[5].T @ grad_x[14]
    grad_x[12] = H(x[12]) * grad_x[13]
    grad_x[11] = w[4].T @ grad_x[12]
    grad_x[10] = grad_x[11].reshape(64,7,7)
    grad_x[9] = (mask_cache[1].reshape(64,7,2,7,2).transpose(0,1,3,2,4) * grad_x[10].reshape(64,7,7,1,1)).transpose(0,1,3,2,4).reshape(64,14,14)
    grad_x[8] = H(x[8]) * grad_x[9]
    grad_x[7] = np.array([back_conv_single_channel(grad_x[8], w[3][:, i]) for i in range(64)])
    grad_x[6] = H(x[6]) * grad_x[7]
    grad_x[5] = np.array([back_conv_single_channel(grad_x[6], w[2][:, i]) for i in range(32)])
    grad_x[4] = (mask_cache[0].reshape(32,14,2,14,2).transpose(0,1,3,2,4) * grad_x[5].reshape(32,14,14,1,1)).transpose(0,1,3,2,4).reshape(32,28,28)
    grad_x[3] = H(x[3]) * grad_x[4]
    grad_x[2] = np.array([back_conv_single_channel(grad_x[3], w[1][:, i]) for i in range(32)])
    grad_x[1] = H(x[1]) * grad_x[2]
    grad_x[0] = None # dont need this
    # for max_pool backprop, alternatively use tiling method (easier, worse)

    grad_b[5] = grad_x[14]
    grad_b[4] = grad_x[12]
    grad_b[3] = np.sum(grad_x[8], axis = (1, 2)).reshape(-1, 1)
    grad_b[2] = np.sum(grad_x[6], axis = (1, 2)).reshape(-1, 1)
    grad_b[1] = np.sum(grad_x[3], axis = (1, 2)).reshape(-1, 1)
    grad_b[0] = np.sum(grad_x[1], axis = (1, 2)).reshape(-1, 1) # size (32, 1)

    grad_w[5] = grad_x[14] @ x[13].T
    grad_w[4] = grad_x[12] @ x[11].T
    
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
        grad_w[0][i] = back_conv_single_kernel(grad_x[1][i], x[0])

    # clean up with matmul (vectorize)

    return grad_w, grad_b




def train(x_train, y_train):
    w = [np.sqrt(2. / (3 * 3)) * np.random.randn(32, 3, 3), np.sqrt(2. / (32 * 3 * 3)) * np.random.randn(32, 32, 3, 3), np.sqrt(2. / (32 * 3 * 3)) * np.random.randn(64, 32, 3, 3), np.sqrt(2. / (64 * 3 * 3)) * np.random.randn(64, 64, 3, 3), np.sqrt(2. / (3136)) * np.random.randn(128, 3136), np.sqrt(2. / (128)) * np.random.randn(10, 128)]
    b = [np.zeros((32, 1)), np.zeros((32, 1)), np.zeros((64, 1)), np.zeros((64, 1)), np.zeros((128, 1)), np.zeros((10, 1))]
    # initialize weights

    for epoch in range(epochs):
        perm = np.random.permutation(train_size)
        x_train = x_train[perm]
        y_train = y_train[perm]

        grad_w_batch = [np.zeros((32, 3, 3)), np.zeros((32, 32, 3, 3)), np.zeros((64, 32, 3, 3)), np.zeros((64, 64, 3, 3)), np.zeros((128, 3136)), np.zeros((10, 128))]
        grad_b_batch = [np.zeros((32, 1)), np.zeros((32, 1)), np.zeros((64, 1)), np.zeros((64, 1)), np.zeros((128, 1)), np.zeros((10, 1))]
        
        for i, (image, target) in tqdm(enumerate(zip(x_train, y_train))):
            x, mask_cache = forward(w, b, image)

            # for j in range(15):
                # print(f'x{j} shape: ', x[j].shape)

            y = one_hot_encode(target)
            grad_w, grad_b = gradient_calculation(w, b, x, y, mask_cache)

            if i % 1 == 0:
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

                grad_w_batch = [np.zeros((32, 3, 3)), np.zeros((32, 32, 3, 3)), np.zeros((64, 32, 3, 3)), np.zeros((64, 64, 3, 3)), np.zeros((128, 3136)), np.zeros((10, 128))]
                grad_b_batch = [np.zeros((32, 1)), np.zeros((32, 1)), np.zeros((64, 1)), np.zeros((64, 1)), np.zeros((128, 1)), np.zeros((10, 1))]

        # process remaining data in last "mini batch"

    return w, b


def test():
    training_data = pd.read_csv("mnist_train.csv", header = None)
    x_train, y_train = training_data.values[:, 1:], training_data.values[:, 0] # [row number, column number]
    x_train_std = x_train.std(axis = 0) + 1e-8
    x_train_mean = x_train.mean(axis = 0)
    x_train = (x_train - x_train_mean) / x_train_std
                        
    testing_data = pd.read_csv("mnist_test.csv", header = None)
    x_test, y_test = testing_data.values[:, 1:], testing_data.values[:, 0]
    x_test = (x_test - x_train_mean) / x_train_std # normalize
    
    w, b = train(x_train, y_train)

    wrong = 0
    for i, (image, target) in enumerate(zip(x_test, y_test)):
        x, _ = forward(w, b, image)
        if(softmax(x[-1]).argmax() != target):
            wrong += 1

    print("epochs: " + str(epochs))
    print("accuracy: " + str(wrong))

    with open("MNIST_CNN_wb.pkl", "wb") as f:
        pickle.dump({"weights": w, "biases": b}, f)

test()



