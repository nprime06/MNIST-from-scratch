import numpy as np

# can take arbitrary dimensions

def softmax(x):
	x_max = np.max(x, axis = 1, keepdims = True)
	e_x = np.exp(x - x_max)
	return e_x / (np.sum(e_x, axis = 1).reshape(-1, 1) + 1e-8)


class Sigmoid:
	def __init__(self):
		self.x = None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, **kwargs):
		self.x = (1 / (1 + np.exp(-1 * in_x)))
		return self.x

	def back(self, grad_out_x, **kwargs):
		grad_in_x = grad_out_x * (self.x * (1 - self.x))

		return grad_in_x


class Tanh:
	def __init__(self):
		self.x = None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, **kwargs):
		self.x = np.tanh(in_x)
		return self.x

	def back(self, grad_out_x, **kwargs):
		grad_in_x = grad_out_x * (1 - (self.x * self.x))
		return grad_in_x