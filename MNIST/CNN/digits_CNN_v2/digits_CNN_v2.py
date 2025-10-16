import sys
sys.path.append('../../../Modules')
import numpy as np
import pandas as pd
import pickle
import time
import idx2numpy as idx
from tqdm import tqdm
from Dense import Dense, Flatten
from ReLU import ReLU
from Sigmoid import softmax
from Reg import BatchNorm, Dropout
from Conv import Conv2d, MaxPool2d
from Optimizer import SGD, Momentum, RMSProp, Adam


class MNIST_CNN():
	def __init__(self, layers):
		self.layers = layers
		self.isTraining = None

	def forward(self, image):
		for layer in self.layers:
			image = layer.forward(image, isTraining = self.isTraining)
		return image

	def back(self, grad):
		for layer in reversed(self.layers):
			grad = layer.back(grad)
		return grad

	def eval(self, x_test, y_test):
		self.isTraining = False
		prediction = np.argmax(self.forward(x_test), axis = 1).astype(np.int32)

		return np.sum((prediction == y_test).astype(int))


def main():
	train_size = 60000
	test_size = 10000
	dtype = np.float32
	eps = dtype(1e-8)

	x_train = idx.convert_from_file('../../datasets/fashion-train-images-idx3-ubyte').reshape(train_size, -1)
	y_train = idx.convert_from_file('../../datasets/fashion-train-labels-idx1-ubyte')
	x_train_var = x_train.var(axis = 0, dtype = dtype)
	x_train_mean = x_train.mean(axis = 0, dtype = dtype)
	x_train = (x_train - x_train_mean) / np.sqrt(x_train_var + eps) # standardized per feature
	y_train = np.eye(10, dtype = dtype)[y_train] # oh encode

	x_test = idx.convert_from_file('../../datasets/fashion-test-images-idx3-ubyte').reshape(test_size, -1)
	y_test = idx.convert_from_file('../../datasets/fashion-test-labels-idx1-ubyte')
	x_test = (x_test - x_train_mean) / np.sqrt(x_train_var + eps)
	
	model = MNIST_CNN( [Conv2d(1, 32, kernel_size = 3, use_bias = False), BatchNorm(dims = (32, 28, 28), alpha = 0.01), ReLU(),
						Conv2d(32, 32, kernel_size = 3, use_bias = False), BatchNorm(dims = (32, 28, 28), alpha = 0.01), ReLU(), MaxPool2d(shape = 2), 
						Conv2d(32, 64, kernel_size = 3, use_bias = False), BatchNorm(dims = (64, 14, 14), alpha = 0.01), ReLU(), 
						Conv2d(64, 64, kernel_size = 3, use_bias = False), BatchNorm(dims = (64, 14, 14), alpha = 0.01), ReLU(), MaxPool2d(shape = 2),
						Flatten(), Dense(3136, 128), ReLU(), Dropout(dropout_rate = 0.5), Dense(128, 10)])
	
	
	'''
	model = MNIST_CNN( [Conv2d(1, 32, kernel_size = 3, use_bias = True), ReLU(),
						Conv2d(32, 32, kernel_size = 3, use_bias = True), ReLU(), MaxPool2d(shape = 2), 
						Conv2d(32, 64, kernel_size = 3, use_bias = True), ReLU(), 
						Conv2d(64, 64, kernel_size = 3, use_bias = True), ReLU(), MaxPool2d(shape = 2),
						Flatten(), Dense(3136, 128), ReLU(), Dropout(dropout_rate = 0.5), Dense(128, 10)])
	'''
	'''
	model = MNIST_CNN( [Conv2d(1, 32, kernel_size = 3, use_bias = False), BatchNorm(dims = (32, 28, 28)), ReLU(), MaxPool2d(shape = 2), 
						Conv2d(32, 64, kernel_size = 3, use_bias = False), BatchNorm(dims = (64, 14, 14)), ReLU(), MaxPool2d(shape = 2),
						Flatten(), Dense(3136, 64), ReLU(), Dropout(dropout_rate = 0.5), Dense(64, 10)])
	'''
	'''
	model = MNIST_CNN( [Conv2d(1, 32, kernel_size = 3, use_bias = True), ReLU(), MaxPool2d(shape = 2), 
						Conv2d(32, 64, kernel_size = 3, use_bias = True), ReLU(), MaxPool2d(shape = 2),
						Flatten(), Dense(3136, 64), ReLU(), Dropout(dropout_rate = 0.5), Dense(64, 10)])
	'''
	'''
	model = MNIST_CNN( [Conv2d(1, 10, kernel_size = 5, stride = 3), ReLU(), MaxPool2d(shape = 2),
						Flatten(), Dense(250, 50), ReLU(), Dropout(dropout_rate = 0.3), Dense(50, 10)])
	'''

	epochs = 30
	batch_size = 32
	batches = train_size // batch_size
	optimizer = Adam(model.layers, learning_rate = 0.002)

	for epoch in tqdm(range(epochs)):
		model.isTraining = True
		perm = np.random.permutation(train_size)
		x_train = x_train[perm]
		y_train = y_train[perm]

		batched_x_train = x_train[:(batches * batch_size)].reshape(batches, batch_size, -1)
		batched_y_train = y_train[:(batches * batch_size)].reshape(batches, batch_size, -1)

		for batch, (image, target) in tqdm(enumerate(zip(batched_x_train, batched_y_train))):

			prediction = model.forward(image.reshape(-1, 1, 28, 28))
			# print(prediction.dtype)

			optimizer.zero_grad()
			model.back((softmax(prediction) - target))
			optimizer.step()

		# eval (sets isTraining = False)
		print(f'Epoch {epoch} test accuracy: {model.eval(x_test.reshape(-1, 1, 28, 28), y_test)}/{test_size}')


if __name__ == '__main__':
	main()
