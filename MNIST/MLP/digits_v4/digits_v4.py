import sys
sys.path.append('../../../Modules')
import numpy as np
import pandas as pd
import idx2numpy as idx
import pickle
import time
from tqdm import tqdm
from Dense import Dense
from ReLU import ReLU
from Sigmoid import softmax, Sigmoid
from Reg import BatchNorm, Dropout
from Optimizer import SGD, Momentum, RMSProp, Adam

# all architecture in MLP
# all training moved to main 


class MNIST_MLP():
	def __init__(self, hidden_sizes, dropout_rate = 0, activation = ReLU, useBatchNorm = False): 
		self.input_size = 784
		self.output_size = 10

		layer_sizes = [self.input_size] + hidden_sizes + [self.output_size]

		# architecture: dense + (batchnorm) + relu (dropout)
		# applies same BN and dropout to every hidden layer
		self.layers = []
		for i in range(len(layer_sizes) - 2):
			self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], use_bias = not useBatchNorm))
			if useBatchNorm:
				self.layers.append(BatchNorm(dims = (layer_sizes[i + 1],)))
			self.layers.append(activation())
			if dropout_rate > 0:
				self.layers.append(Dropout(dropout_rate = dropout_rate))
		self.layers.append(Dense(layer_sizes[-2], layer_sizes[-1]))

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

	# training_data = pd.read_csv('../../datasets/mnist_train.csv', header = None)
	# x_train, y_train = training_data.values[:, 1:].astype(dtype), training_data.values[:, 0].astype(np.int32) # [row number, column number]
	
	# testing_data = pd.read_csv('../../datasets/mnist_test.csv', header = None)
	# x_test, y_test = testing_data.values[:, 1:].astype(dtype), testing_data.values[:, 0].astype(np.int32)

	x_train = idx.convert_from_file('../../datasets/train-images-idx3-ubyte').reshape(train_size, -1)
	y_train = idx.convert_from_file('../../datasets/train-labels-idx1-ubyte')
	x_train_var = x_train.var(axis = 0, dtype = dtype)
	x_train_mean = x_train.mean(axis = 0, dtype = dtype)
	x_train = (x_train - x_train_mean) / np.sqrt(x_train_var + eps) # standardized per feature
	y_train = np.eye(10, dtype = dtype)[y_train] # oh encode

	x_test = idx.convert_from_file('../../datasets/test-images-idx3-ubyte').reshape(test_size, -1)
	y_test = idx.convert_from_file('../../datasets/test-labels-idx1-ubyte')
	x_test = (x_test - x_train_mean) / np.sqrt(x_train_var + eps)

	model = MNIST_MLP([256,128], 
						dropout_rate = 0, 
						useBatchNorm = True, 
						activation = ReLU)


	# training hyperparameters
	epochs = 100
	batch_size = 64
	batches = train_size // batch_size
	optimizer = SGD(model.layers, learning_rate = 0.01)

	# training loop

	for epoch in tqdm(range(epochs)):
		model.isTraining = True
		perm = np.random.permutation(train_size)
		x_train = x_train[perm]
		y_train = y_train[perm]

		batched_x_train = x_train[:(batches * batch_size)].reshape(batches, batch_size, -1)
		batched_y_train = y_train[:(batches * batch_size)].reshape(batches, batch_size, -1)

		for batch, (image, target) in tqdm(enumerate(zip(batched_x_train, batched_y_train))):

			prediction = model.forward(image)
			# print(prediction.dtype)

			optimizer.zero_grad()
			model.back((softmax(prediction) - target))
			optimizer.step()

		# eval (sets isTraining = False)
		print(f'Epoch {epoch} test accuracy: {model.eval(x_test, y_test)}/{test_size}')
		


if __name__ == '__main__':
	main()