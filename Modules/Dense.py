import numpy as np

# Dense assumes flat, ie. in dims (N, fan_in) and out dims (N, fan_out) 

class Dense:
	def __init__(self, fan_in, fan_out, init_method = 'He', use_bias = True, dtype = np.float32):
		self.use_bias = use_bias
		self.dtype = dtype

		if init_method == 'He':
			self.w = ((np.sqrt(2. / fan_in)) * np.random.randn(fan_out, fan_in)).astype(self.dtype)
		elif init_method == 'Xa': 
			self.w = ((np.sqrt(2./ (fan_in + fan_out))) * np.random.randn(fan_out, fan_in)).astype(self.dtype)
		self.grad_w = np.zeros_like(self.w)

		if self.use_bias:
			self.b = np.zeros((1, fan_out), dtype = self.dtype)
			self.grad_b = np.zeros_like(self.b)
		else:
			self.b = None
			self.grad_b = None

		self.out_x = None

	def parameters(self):
		return {'params': [self.w, self.b], 'grads': [self.grad_w, self.grad_b]}

	def forward(self, in_x, isTraining, **kwargs):		
		self.in_x = in_x
		out_x = self.in_x @ self.w.T

		if self.use_bias:
			out_x = out_x + self.b

		return out_x
		
		'''
		if not isTraining:
			if self.use_bias:
				return in_x @ self.w.T + self.b
			else:
				return in_x @ self.w.T

		self.in_x = in_x

		if self.out_x is not None:
			self.out_x.fill(0)
			self.out_x += self.in_x @ self.w.T
		else:
			self.out_x = self.in_x @ self.w.T

		if self.use_bias:
			self.out_x += self.b

		return self.out_x'''

	def back(self, grad_out_x, isTraining = True, **kwargs): # returns downstream grad, sets param grads
		N, _ = grad_out_x.shape

		if isTraining:
			self.grad_w += ((grad_out_x.T @ self.in_x) / N)
			if self.use_bias:
				self.grad_b += np.mean(grad_out_x, axis = 0, keepdims = True, dtype = self.dtype)

		return grad_out_x @ self.w

class Flatten:
	def __init__(self):
		self.batch_size = None
		self.dims_in = None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, **kwargs):
		self.batch_size, *self.dims_in = in_x.shape
		return in_x.reshape(self.batch_size, -1)

	def back(self, grad_out_x, **kwargs):
		return grad_out_x.reshape(self.batch_size, *self.dims_in)

