import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


def relu(x):
	return np.maximum(0, x)

def H(x):
	return (x > 0).astype(int)

def leakyrelu(x, alpha = 0.2):
	return np.maximum(alpha * x, x)

def leakyH(x, alpha = 0.2):
	return H(x) * (1 - alpha) + alpha

def sigmoid(x):
	return 1 / (1 + np.exp(-1 * x))

def dsigmoid(x):
	return x * (1 - x)

def tanh(x):
	return np.tanh(x)

def dtanh(x):
	return 1 - x * x


# architecture hyperparams
neurons_gen = [100, 128, 256, 512, 784]
neurons_dis = [784, 256, 128, 1]

# fixed
layers_gen = len(neurons_gen)
activations_gen = (layers_gen - 2) * [relu] + [tanh] # [relu, relu, relu, tanh]
dactivations_gen = (layers_gen - 2) * [H] + [dtanh]

layers_dis = len(neurons_dis)
activations_dis = (layers_dis - 2) * [leakyrelu] + [sigmoid] # [lrelu, lrelu, sig]
dactivations_dis = (layers_dis - 2) * [leakyH] + [dsigmoid]


# training hyperparams (editable)
epochs = 200
batch_size = 64
discriminator_ratio = 2
learning_rate = 0.001

# other (fixed)
train_size = 60000
cycles = train_size // (discriminator_ratio * batch_size) # 60000 // (6 * 32) = 312


def forward_gen(w_gen, b_gen, N):
	z = [None] * layers_gen
	z[0] = np.random.randn(N, neurons_gen[0])

	for i in range(layers_gen - 1):
		z[i + 1] = activations_gen[i](z[i] @ w_gen[i].T + b_gen[i].reshape(1, -1))

	return z


def forward_dis(w_dis, b_dis, image): # image normalized to [-1, 1]
	x = [None] * layers_dis
	x[0] = image # shape: (N, -1)

	for i in range(layers_dis - 1):
		x[i + 1] = activations_dis[i](x[i] @ w_dis[i].T + b_dis[i].reshape(1, -1))

	return x


def grad_calculation_gen(w_gen, b_gen, w_dis, b_dis):
	grad_x, grad_z, grad_w_gen, grad_b_gen = [None] * layers_dis, [None] * layers_gen, [None] * (layers_gen - 1), [None] * (layers_gen - 1)
	# grad_z: dL / dz
	# grad_x: dx[-1] / dx

	z = forward_gen(w_gen, b_gen, batch_size)
	x = forward_dis(w_dis, b_dis, z[-1])

	grad_x[-1] = np.ones((batch_size, 1))
	for i in reversed(range(layers_dis - 1)):
		grad_x[i] = (dactivations_dis[i](x[i + 1]) * grad_x[i + 1]) @ w_dis[i]

	grad_z[-1] = (-1 / (x[-1] + 1e-6)) * grad_x[0]
	for i in reversed(range(layers_gen - 1)):
		grad_z[i] = (dactivations_gen[i](z[i + 1]) * grad_z[i + 1]) @ w_gen[i]

	for i in range(layers_gen - 1):
		grad_w_gen[i] = (dactivations_gen[i](z[i + 1]) * grad_z[i + 1]).T @ z[i]
		grad_b_gen[i] = np.sum((dactivations_gen[i](z[i + 1]) * grad_z[i + 1]), axis = 0)

	return grad_w_gen, grad_b_gen


def grad_calculation_dis(w_gen, b_gen, w_dis, b_dis, image): # add dropout
	grad_x, grad_w_dis, grad_b_dis = [None] * layers_dis, [None] * (layers_dis - 1), [None] * (layers_dis - 1)

	if image is None: 
		z = forward_gen(w_gen, b_gen, batch_size)
		x = forward_dis(w_dis, b_dis, z[-1]) # fake -> generate 
		grad_x[-1] = 1 / (0.9 - x[-1] + 1e-6) # use -log(1-D(G(z))) 
		# label smoothing
	else: 
		x = forward_dis(w_dis, b_dis, image)
		grad_x[-1] = (-1 / (x[-1] + 1e-6)) # real -> use -log(D(x))

	for i in reversed(range(layers_dis - 1)):
		grad_x[i] = (dactivations_dis[i](x[i + 1]) * grad_x[i + 1]) @ w_dis[i]

	for i in range(layers_dis - 1):
		grad_w_dis[i] = (dactivations_dis[i](x[i + 1]) * grad_x[i + 1]).T @ x[i]
		grad_b_dis[i] = np.sum((dactivations_dis[i](x[i + 1]) * grad_x[i + 1]), axis = 0)

	return grad_w_dis, grad_b_dis


def train(training_images):
	# main training loop	

	w_gen = [(np.sqrt(2.0 / neurons_gen[i])) * np.random.randn(neurons_gen[i + 1], neurons_gen[i]) for i in range(layers_gen - 2)] + [2 * (np.sqrt(6. / (neurons_gen[-1] + neurons_gen[-2]))) * np.random.rand(neurons_gen[-1], neurons_gen[-2]) - (np.sqrt(6. / (neurons_gen[-1] + neurons_gen[-2])))]
	b_gen = [np.zeros((neurons_gen[i + 1], )) for i in range(layers_gen - 1)]
	w_dis = [(np.sqrt(2.0 / neurons_dis[i])) * np.random.randn(neurons_dis[i + 1], neurons_dis[i]) for i in range(layers_dis - 2)] + [2 * (np.sqrt(6. / (neurons_dis[-1] + neurons_dis[-2]))) * np.random.rand(neurons_dis[-1], neurons_dis[-2]) - (np.sqrt(6. / (neurons_dis[-1] + neurons_dis[-2])))]
	b_dis = [np.zeros((neurons_dis[i + 1], )) for i in range(layers_dis - 1)]

	for epoch in tqdm(range(epochs)):
		perm = np.random.permutation(train_size)
		training_images = training_images[perm]

		batched_training_images = training_images[:(cycles * discriminator_ratio * batch_size)].reshape(cycles, discriminator_ratio, batch_size, -1)

		for cycle, cycled_discriminator_images in enumerate(batched_training_images):
			for batch, image in enumerate(cycled_discriminator_images):

				real_grad_w_dis, real_grad_b_dis = grad_calculation_dis(w_gen, b_gen, w_dis, b_dis, image)
				fake_grad_w_dis, fake_grad_b_dis = grad_calculation_dis(w_gen, b_gen, w_dis, b_dis, None)

				for i in range(layers_dis - 1):
					w_dis[i] -= learning_rate * (real_grad_w_dis[i] / batch_size)
					w_dis[i] -= learning_rate * (fake_grad_w_dis[i] / batch_size)
					b_dis[i] -= learning_rate * (real_grad_b_dis[i] / batch_size)
					b_dis[i] -= learning_rate * (fake_grad_b_dis[i] / batch_size)
				# discriminator learning

			grad_w_gen, grad_b_gen = grad_calculation_gen(w_gen, b_gen, w_dis, b_dis)

			for i in range(layers_gen - 1):
				w_gen[i] -= learning_rate * (grad_w_gen[i] / batch_size)
				b_gen[i] -= learning_rate * (grad_b_gen[i] / batch_size)
			# generator learning

		if (epoch + 1) % 3 == 0: # troubleshooting

			z_fake = forward_gen(w_gen, b_gen, 1)
			x_fake = forward_dis(w_dis, b_dis, z_fake[-1])
			x_real = forward_dis(w_dis, b_dis, training_images[-1].reshape(1, -1))

			print("D(x_real): ", x_real[-1][0][0], ", D(G(z)): ", x_fake[-1][0][0])

			# better code this so its average of a batch, and also calculate the loss

			if (epoch + 1) % 4 == 0:

				plt.imshow((z_fake[-1][0].reshape(28, 28) + 1) / 2, cmap='gray')
				plt.axis('off')  # Hide axes
				plt.title(f"Epoch {epoch}")
				plt.savefig(f"gan_samples_v0/epoch_{epoch}.png")
				plt.close()
		
	return w_gen, b_gen, w_dis, b_dis

def test():
	with open(f"MNIST_GAN_wb.pkl", "rb") as f:
		unpickle = pickle.load(f)
		w_gen = unpickle["weights_gen"]
		b_gen = unpickle["biases_gen"]
	
	z = forward_gen(w_gen, b_gen, 1)

	image = (z[-1][0].reshape(28, 28) + 1) / 2

	plt.imshow(image, cmap='gray')
	plt.axis('off')  # Hide axes
	plt.title("28x28 Image")


def main():
	training_data = pd.read_csv('/Users/william/Desktop/MNIST/mnist_train.csv', header = None)
	training_images, _ = 2 * (training_data.values[:, 1:] / 255) - 1, training_data.values[:, 0] # normalize

	w_gen, b_gen, w_dis, b_dis = train(training_images)

	with open("MNIST_GAN_wb.pkl", "wb") as f:
		pickle.dump({"weights_gen": w_gen, "biases_gen": b_gen, "weights_dis": w_dis, "biases_dis": b_dis}, f) 
		# pickle neurons and activations later


if __name__ == "__main__":
	main()

