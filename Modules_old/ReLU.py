import numpy as np

# can take arbitrary dimensions

class ReLU:
	def __init__(self):
		self.in_x = None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, **kwargs):
		self.in_x = in_x
		return np.maximum(self.in_x, 0)

	def back(self, grad_out_x, **kwargs):
		return grad_out_x * (self.in_x > 0).astype(float)

class LeakyReLU:
	def __init__(self, alpha):
		self.alpha = alpha
		self.in_x = None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, **kwargs):
		self.in_x = in_x
		return np.maximum(self.in_x, self.alpha * self.in_x)

	def back(self, grad_out_x, **kwargs):
		return grad_out_x * ((self.in_x > 0).astype(float) * (1 - self.alpha) + self.alpha) 
