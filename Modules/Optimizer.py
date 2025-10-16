import numpy as np
import math

# Takes in a mean gradient (never deals with batch size)
# Note: if you freeze a parameter, make sure its grad is None

# Add adagrad

class SGD:
	def __init__(self, layers, learning_rate, dtype = np.float32):
		self.dtype = dtype
		self.learning_rate = self.dtype(learning_rate)

		self.params = [param for layer in layers for param in layer.parameters()['params'] if param is not None]
		self.grads = [grad for layer in layers for grad in layer.parameters()['grads'] if grad is not None]

	def step(self): 
		for (param, grad) in zip(self.params, self.grads):
			if grad is None: # double check that is theoretically not necessary
				continue
			param -= self.learning_rate * grad

	def zero_grad(self): 
		for grad in self.grads:
			if grad is None: 
				continue
			grad.fill(0)


class Momentum:
	def __init__(self, layers, learning_rate, gamma = 0.9, dtype = np.float32):
		self.dtype = dtype
		self.learning_rate = self.dtype(learning_rate)
		self.gamma = self.dtype(gamma)

		self.params = [param for layer in layers for param in layer.parameters()['params'] if param is not None]
		self.grads = [grad for layer in layers for grad in layer.parameters()['grads'] if grad is not None]

		self.momentum = [np.zeros_like(grad) for grad in self.grads]

	def step(self):
		for i, (param, grad) in enumerate(zip(self.params, self.grads)):
			if grad is None:
				continue

			#self.momentum[i] *= self.gamma
			#self.momentum[i] += self.learning_rate * grad
			self.momentum[i] = self.gamma * self.momentum[i] + self.learning_rate * grad
			param -= self.momentum[i]


	def zero_grad(self): 
		for grad in self.grads:
			if grad is None: 
				continue
			grad.fill(0)


class RMSProp:
	def __init__(self, layers, learning_rate, beta = 0.9, eps = 1e-8, dtype = np.float32):
		self.dtype = dtype
		self.learning_rate = self.dtype(learning_rate)
		self.beta = self.dtype(beta)
		self.eps = self.dtype(eps)

		self.params = [param for layer in layers for param in layer.parameters()['params'] if param is not None]
		self.grads = [grad for layer in layers for grad in layer.parameters()['grads'] if grad is not None]

		self.v = [np.zeros_like(grad) for grad in self.grads]

	def step(self):
		for i, (param, grad) in enumerate(zip(self.params, self.grads)):
			if grad is None:
				continue

			#self.v[i] *= self.beta
			#self.v[i] += (1 - self.beta) * np.power(grad, 2)
			self.v[i] = self.beta * self.v[i] + (1 - self.beta) * np.power(grad, 2)
			param -= self.learning_rate * grad / np.sqrt(self.v[i] + self.eps)
	
	def zero_grad(self): 
		for grad in self.grads:
			if grad is None: 
				continue
			grad.fill(0)


class Adam:
	def __init__(self, layers, learning_rate, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, dtype = np.float32):
		self.dtype = dtype
		self.learning_rate = self.dtype(learning_rate)
		self.beta1 = self.dtype(beta1)
		self.beta2 = self.dtype(beta2)
		self.eps = self.dtype(eps)

		self.params = [param for layer in layers for param in layer.parameters()['params'] if param is not None]
		self.grads = [grad for layer in layers for grad in layer.parameters()['grads'] if grad is not None]

		self.moment1 = [np.zeros_like(grad) for grad in self.grads]
		self.moment2 = [np.zeros_like(grad) for grad in self.grads]

		self.t = 0

	def step(self):
		self.t += 1

		for i, (param, grad) in enumerate(zip(self.params, self.grads)):
			if grad is None:
				continue

			'''self.moment1[i] *= self.beta1
			self.moment2[i] *= self.beta2
			self.moment1[i] += (1 - self.beta1) * grad
			self.moment2[i] += (1 - self.beta2) * np.power(grad, 2)'''
			self.moment1[i] = self.beta1 * self.moment1[i] + (1 - self.beta1) * grad
			self.moment2[i] = self.beta2 * self.moment2[i] + (1 - self.beta2) * np.power(grad, 2)
			# m1hat = self.moment1[i] / (1 - self.dtype(math.pow(self.beta1, self.t)))
			# m2hat = self.moment2[i] / (1 - self.dtype(math.pow(self.beta2, self.t)))
			#alpha_t = self.learning_rate

			alpha_t = self.learning_rate * self.dtype(math.sqrt(1 - math.pow(self.beta2, self.t)) / (1 - math.pow(self.beta1, self.t))) # see Kingma et al.
			param -= alpha_t * self.moment1[i] / (np.sqrt(self.moment2[i]) + self.eps)

	
	def zero_grad(self): 
		for grad in self.grads:
			if grad is None: 
				continue
			grad.fill(0)