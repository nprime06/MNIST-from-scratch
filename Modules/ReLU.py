import numpy as np

# takes arbitrary dimensions

class ReLU:
	def __init__(self, dtype = np.float32):
		self.dtype = dtype
		self.in_x = None
		self.out_x = None
		self.grad_in_x = None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, isTraining, **kwargs): 
		if not isTraining: # no need to store in_x
			return np.maximum(in_x, 0)

		self.in_x = in_x
		return np.maximum(self.in_x, 0) ###
		
		'''if self.out_x is not None:
			self.out_x.fill(0)
		else:
			self.out_x = np.zeros_like(in_x)

		self.out_x += np.maximum(self.in_x, 0)
		return self.out_x'''

	def back(self, grad_out_x, **kwargs):
		return grad_out_x * (self.in_x > 0).astype(self.dtype) ###
		'''if self.grad_in_x is not None:
			self.grad_in_x.fill(0)
		else:
			self.grad_in_x = np.zeros_like(grad_out_x)

		self.grad_in_x += grad_out_x * (self.in_x > 0).astype(self.dtype)
		return self.grad_in_x'''

class LeakyReLU:
	def __init__(self, alpha, dtype = np.float32):
		self.dtype = dtype
		self.alpha = self.dtype(alpha)
		self.in_x = None
		self.out_x = None
		self.grad_in_x = None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, isTraining, **kwargs):
		if not isTraining: 
			return np.maximum(in_x, self.alpha * in_x)

		self.in_x = in_x
		return np.maximum(self.in_x, self.alpha * self.in_x) ###

		'''
		if self.out_x is not None:
			self.out_x.fill(0)
		else:
			self.out_x = np.zeros_like(in_x)

		self.out_x += np.maximum(self.in_x, self.alpha * self.in_x)
		return self.out_x'''

	def back(self, grad_out_x, **kwargs):
		return grad_out_x * ((self.in_x > 0).astype(self.dtype) * (1 - self.alpha) + self.alpha) ###
		'''if self.grad_in_x is not None:
			self.grad_in_x.fill(0)
		else:
			self.grad_in_x = np.zeros_like(grad_out_x)

		self.grad_in_x += grad_out_x * ((self.in_x > 0).astype(self.dtype) * (1 - self.alpha) + self.alpha) 
		return self.grad_in_x'''
