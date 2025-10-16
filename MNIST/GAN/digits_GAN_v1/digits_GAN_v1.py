import sys
import os
sys.path.append('../../Modules_old')
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from Dense import Dense
from ReLU import ReLU, LeakyReLU
from Sigmoid import Sigmoid, Tanh
from Reg import BatchNorm, Dropout

##### generator architecture #####
# noise N(0, 1) -> dense -> batch norm -> relu 
# 				-> ...
# 				-> dense -> tanh

##### discriminator architecture #####
# image -> dense -> leaky relu (0.2) + dropout (0.3)
# 		-> ...
# image -> dense -> sigmoid


class GeneratorMLP():
	def __init__(self, latent_noise_size, hidden_sizes):
		self.latent_noise_size = latent_noise_size
		self.output_size = 784

		layer_sizes = [self.latent_noise_size] + hidden_sizes + [self.output_size]

		# architecture: dense + batch norm + relu (tanh output)
		self.layers = []
		for i in range(len(layer_sizes) - 2):
			self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], use_bias = False))
			self.layers.append(BatchNorm())
			self.layers.append(ReLU())
		self.layers.append(Dense(layer_sizes[-2], layer_sizes[-1], init_method = 'Xa'))
		self.layers.append(Tanh())

		self.isTraining = False


	def forward(self, batch_size):
		image = np.random.randn(batch_size, self.latent_noise_size)

		for layer in self.layers:
			image = layer.forward(image, isTraining = self.isTraining)

		return image

	def back(self, grad, learning_rate):
		for layer in reversed(self.layers):
			grad = layer.back(grad, learning_rate = learning_rate, isTraining = self.isTraining)

		return grad


class DiscriminatorMLP():
	def __init__(self, hidden_sizes):
		self.input_size = 784
		self.output_size = 1

		layer_sizes = [self.input_size] + hidden_sizes + [self.output_size]

		# architecture: dense + leaky relu (sigmoid output) + dropout
		self.layers = []
		for i in range(len(layer_sizes) - 2):
			self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1]))
			self.layers.append(LeakyReLU(alpha = 0.2))
			if i == 0:
				self.layers.append(Dropout(dropout_rate = 0.3))
		self.layers.append(Dense(layer_sizes[-2], layer_sizes[-1], init_method = 'Xa'))
		self.layers.append(Sigmoid())

		self.isTraining = False


	def forward(self, image):
		for layer in self.layers:
			image = layer.forward(image, isTraining = self.isTraining)

		return image


	def back(self, grad, learning_rate):
		for layer in reversed(self.layers):
			grad = layer.back(grad, learning_rate = learning_rate, isTraining = self.isTraining)

		return grad


def main():
	folder_id = 0
	while True:
		try:
			os.makedirs(f'samples/gan_samples_v1_{folder_id}')
		except FileExistsError:
			folder_id += 1
			continue
		break

	train_size = 60000
	training_data = pd.read_csv('/Users/william/Desktop/MNIST/mnist_train.csv', header = None)
	training_images, labels = 2 * (training_data.values[:, 1:] / 255) - 1, training_data.values[:, 0] # normalize to [-1, 1]

	Gen = GeneratorMLP(latent_noise_size = 100, hidden_sizes = [256, 512])
	Dis = DiscriminatorMLP(hidden_sizes = [512, 256])


	# training params:
	epochs = 200
	batch_size = 32
	dis_learning_rate = 0.001
	gen_learning_rate = 0.001
	dis_per_cycle = 1
	gen_per_cycle = 1
	dis_real_label_smooth = 0.1
	dis_fake_label_smooth = 0
	eps = 1e-6

	cycles = train_size // (batch_size * dis_per_cycle)
	
	for epoch in tqdm(range(epochs)):
		perm = np.random.permutation(train_size)
		training_images = training_images[perm]

		batched_training_images = training_images[:(cycles * batch_size * dis_per_cycle)].reshape(cycles, dis_per_cycle, batch_size, -1)

		for cycle, cycled_discriminator_images in tqdm(enumerate(batched_training_images)):
		
			Dis.isTraining, Gen.isTraining = True, False
			for batch, image in enumerate(cycled_discriminator_images):
				concatenated_output = Dis.forward(np.concatenate((Gen.forward(batch_size), image), axis = 0))
				fake_grad_out = (1 - dis_fake_label_smooth) * (1 / (1 - concatenated_output[:batch_size] + eps)) + (dis_fake_label_smooth) * (-1 / (concatenated_output[:batch_size] + eps))
				real_grad_out = (dis_real_label_smooth) * (1 / (1 - concatenated_output[batch_size:] + eps)) + (1 - dis_real_label_smooth) * (-1 / (concatenated_output[batch_size:] + eps))
				Dis.back(np.concatenate((fake_grad_out, real_grad_out), axis = 0), learning_rate = dis_learning_rate)

				#Dis.back(1 / (1 - Dis.forward(Gen.forward(batch_size)) + eps), learning_rate = dis_learning_rate)
				#Dis.back(-1 / (Dis.forward(image) + eps), learning_rate = dis_learning_rate)

			Gen.isTraining, Dis.isTraining = True, False
			for _ in range(gen_per_cycle):
				Gen.back((-1 / (Dis.forward(Gen.forward(batch_size)) + eps)) * Dis.back(np.ones((batch_size, 1)), learning_rate = 0), learning_rate = gen_learning_rate)
				# sketchy pemdas :)

		if epoch % 10 == 0: 
			Gen.isTraining, Dis.isTraining = False, False
			Gz = Gen.forward(16)
			DGz = Dis.forward(Gz)
			x = training_images[:16]
			Dx = Dis.forward(x)

			G_loss = - np.ravel(np.mean(np.log(DGz)))
			D_loss = - np.ravel(np.mean(np.log(Dx) + np.log(1 - DGz)))

			print(f"epoch: {epoch}, G_loss: {G_loss}, D_loss: {D_loss}")
			print("Dx, DGz:")
			print(np.ravel(Dx), np.ravel(DGz))

			gallery = (Gz.reshape(16, 28, 28) + 1) / 2
			fig, axes = plt.subplots(4, 4, figsize=(6, 6))
			plt.suptitle(f"N: {batch_size}; lr: {dis_learning_rate}/{gen_learning_rate}; D/G: {dis_per_cycle}/{gen_per_cycle}")

			# Flatten axes for easy iteration
			axes = axes.flatten()
			for i in range(16):
			    ax = axes[i]
			    ax.imshow(gallery[i], cmap='gray')
			    ax.axis('off')  # Hide axis

			plt.tight_layout()
			plt.savefig(f"samples/gan_samples_v1_{folder_id}/epoch_{epoch}.png", dpi=300)
			plt.close()

	# loop breakdown: 
	# epoch: loop over all training data
		# cycle: train Dis k_dis times, train Gen k_gen times
			# batch



if __name__ == "__main__":
    main()

