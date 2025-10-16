import numpy as np

# Dense assumes flat, ie. in dims (N, fan_in) and out dims (N, fan_out) 

class Dense:
	def __init__(self, fan_in, fan_out, init_method = 'He', use_bias = True, dtype = np.float32):
		self.fan_in = fan_in
		self.fan_out = fan_out
		self.use_bias = use_bias

		if init_method == 'He':
			self.w = ((np.sqrt(2. / fan_in)) * np.random.randn(fan_out, fan_in)).astype(dtype)
		elif init_method == 'Xa': 
			self.w = ((np.sqrt(2 / (fan_in + fan_out))) * np.random.randn(fan_out, fan_in)).astype(dtype)
		self.grad_w = np.zeros_like(self.w)

		if self.use_bias:
			self.b = np.zeros((1, fan_out), dtype = dtype)
			self.grad_b = np.zeros_like(self.b)
		else:
			self.b = None
			self.grad_b = None


	def parameters(self):
		return {'params': [self.w, self.b], 'grads': [self.grad_w, self.grad_b]}



	def forward(self, in_x, **kwargs):
		self.in_x = in_x
		out_x = self.in_x @ self.w.T

		if self.use_bias:
			out_x = out_x + self.b

		return out_x

	def back(self, grad_out_x, learning_rate, isTraining = True, **kwargs): # returns downstream grad, sets param grads
		N, _ = grad_out_x.shape

		if isTraining:
			self.w -= learning_rate * ((grad_out_x.T @ self.in_x) / N)
			if self.use_bias:
				self.b -= learning_rate * np.mean(grad_out_x, axis = 0, keepdims = True)

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

