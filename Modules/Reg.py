import numpy as np

# penis

class Dropout:
	def __init__(self, dropout_rate, dtype = np.float32):
		self.dtype = dtype
		self.dropout_rate = self.dtype(dropout_rate)
		self.mask = None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, isTraining = True, **kwargs):
		if isTraining:
			self.mask = (np.random.rand(*in_x.shape) > self.dropout_rate).astype(self.dtype) / (1.0 - self.dropout_rate)
			return in_x * self.mask
		else:
			return in_x

	def back(self, grad_out, isTraining = True, **kwargs):
		if isTraining:
			return grad_out * self.mask
		else:
			return grad_out


# assumes first dimension is batch size
# if flat, batch norm by feature
# if conv, batch norm by channel

class BatchNorm: # dims exclude N, eg. dims = (C, h, w)
	def __init__(self, dims, alpha = 0.1, eps = 1e-3, dtype = np.float32):
		self.dtype = dtype
		self.axis = tuple(i for i in range(len(dims) + 1) if i != 1)
		dims_1 = [1 for _ in dims]
		dims_1[0] = dims[0]
		dims_1 = tuple(dims_1)

		self.gamma = np.ones((1, *dims_1), dtype = self.dtype)
		self.beta = np.zeros((1, *dims_1), dtype = self.dtype)
		self.grad_gamma = np.zeros_like(self.gamma)
		self.grad_beta = np.zeros_like(self.beta)

		self.mu_curr = None
		self.var_curr = None

		self.alpha = self.dtype(alpha)
		self.mu_moving = np.zeros((1, *dims_1), dtype = self.dtype)
		self.var_moving = np.zeros((1, *dims_1), dtype = self.dtype)

		self.in_x_hat = None
		# self.out_x = None
		self.eps = self.dtype(eps)


	def parameters(self):
		return {'params': [self.gamma, self.beta], 'grads': [self.grad_gamma, self.grad_beta]}


	def forward(self, in_x, isTraining = True):
		'''
		if isTraining:
			if self.mu_curr is not None: 
				self.mu_curr.fill(0)
				self.var_curr.fill(0)
				self.in_x_hat.fill(0)
				self.out_x.fill(0)

				self.mu_curr += np.mean(in_x, axis = self.axis, keepdims = True, dtype = self.dtype)
				self.var_curr += np.var(in_x, axis = self.axis, keepdims = True, dtype = self.dtype)
				self.in_x_hat += (in_x - self.mu_curr) / (np.sqrt(self.var_curr + self.eps))
				self.out_x += self.gamma * self.in_x_hat + self.beta

				self.mu_moving *= (1 - self.alpha)
				self.var_moving *= (1 - self.alpha)
				self.mu_moving += self.alpha * self.mu_curr
				self.var_moving += self.alpha * self.var_curr

			else: 
				self.mu_curr = np.mean(in_x, axis = self.axis, keepdims = True, dtype = self.dtype) # preserves axis 1 dimensions (features/channels)
				self.var_curr = np.var(in_x, axis = self.axis, keepdims = True, dtype = self.dtype)

				self.in_x_hat = (in_x - self.mu_curr) / (np.sqrt(self.var_curr + self.eps))
				self.out_x = self.gamma * self.in_x_hat + self.beta

				self.mu_moving += self.alpha * self.mu_curr
				self.var_moving += self.alpha * self.var_curr

			return self.out_x

		else: 
			return self.gamma * ((in_x - self.mu_moving) / (np.sqrt(self.var_moving + self.eps))) + self.beta
		'''

		if isTraining:
			self.mu_curr = np.mean(in_x, axis = self.axis, keepdims = True, dtype = self.dtype) # preserves axis 1 dimensions (features/channels)
			self.var_curr = np.var(in_x, axis = self.axis, keepdims = True, dtype = self.dtype)

			self.in_x_hat = (in_x - self.mu_curr) / (np.sqrt(self.var_curr + self.eps))
			out_x = self.gamma * self.in_x_hat + self.beta

			self.mu_moving = (1 - self.alpha) * self.mu_moving + self.alpha * self.mu_curr
			self.var_moving = (1 - self.alpha) * self.var_moving + self.alpha * self.var_curr

		else:
			self.in_x_hat = (in_x - self.mu_moving) / (np.sqrt(self.var_moving + self.eps))
			out_x = self.gamma * self.in_x_hat + self.beta
			
		return out_x
		

	def back(self, grad_out_x, isTraining = True, **kwargs):
		if isTraining:
			self.grad_gamma += np.mean(self.in_x_hat * grad_out_x, axis = self.axis, keepdims = True, dtype = self.dtype)
			self.grad_beta += np.mean(grad_out_x, axis = self.axis, keepdims = True, dtype = self.dtype)
		# else:
			# isTraining = false

		grad_in_x_hat = self.gamma * grad_out_x
		inv_std = 1. / (np.sqrt(self.var_curr + self.eps))
		sum_grad_in_x_hat = np.mean(grad_in_x_hat, axis = self.axis, keepdims = True, dtype = self.dtype)
		sum_grad_in_x_hat_in_x_hat = np.mean(grad_in_x_hat * self.in_x_hat, axis = self.axis, keepdims = True, dtype = self.dtype)
		grad_in_x = inv_std * (
			grad_in_x_hat
			- sum_grad_in_x_hat
			- self.in_x_hat * sum_grad_in_x_hat_in_x_hat
			)

		'''grad_in_x = (1. / N_group) * inv_std * (
			N_group * grad_in_x_hat
			- sum_grad_in_x_hat
			- self.in_x_hat * sum_grad_in_x_hat_in_x_hat
			)'''

		return grad_in_x

		# pls add support for istraining
