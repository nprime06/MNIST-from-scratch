import numpy as np

class Dropout:
	def __init__(self, dropout_rate):
		self.dropout_rate = dropout_rate
		self.mask = None

	def forward(self, in_x, isTraining = True, **kwargs):
		if isTraining:
			self.mask = (np.random.rand(*in_x.shape) > self.dropout_rate).astype(float) / (1.0 - self.dropout_rate)
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

class BatchNorm:
	def __init__(self, alpha = 0.1):
		self.gamma = None
		self.beta = None
		self.mu_curr = None
		self.var_curr = None

		self.alpha = alpha # momentum
		self.mu_moving = None
		self.var_moving = None

		self.in_x = None
		self.in_x_hat = None


	def forward(self, in_x, isTraining = True):
		self.in_x = in_x
		_, C, *dim = self.in_x.shape
		axis = tuple([i for i in range(len(in_x.shape)) if i != 1]) # Axes over which mean/var are computed

		if self.gamma is None: 
			shape = (1, C, *([1] * len(dim)))
			self.gamma = np.ones(shape, dtype = in_x.dtype)
			self.beta = np.zeros(shape, dtype = in_x.dtype)
			self.mu_moving = np.zeros(shape, dtype = in_x.dtype)
			self.var_moving = np.zeros(shape, dtype = in_x.dtype)

		if isTraining:
			self.mu_curr = np.mean(in_x, axis = axis, keepdims = True)
			self.var_curr = np.var(in_x, axis = axis, keepdims = True)

			self.in_x_hat = (self.in_x - self.mu_curr) / (np.sqrt(self.var_curr + 1e-3))
			out_x = self.gamma * self.in_x_hat + self.beta

			self.mu_moving = (1 - self.alpha) * self.mu_moving + self.alpha * self.mu_curr
			self.var_moving = (1 - self.alpha) * self.var_moving + self.alpha * self.var_curr

		else: # Inference/Evaluation mode
			self.in_x_hat = (self.in_x - self.mu_moving) / (np.sqrt(self.var_moving + 1e-3)) # Uses running averages
			out_x = self.gamma * self.in_x_hat + self.beta
			
		return out_x


	def back(self, grad_out_x, learning_rate, isTraining = True, **kwargs):
		if not isTraining: 
			return grad_out_x # or raise an error

		N, C, *dim = grad_out_x.shape
		axis = tuple([i for i in range(len(grad_out_x.shape)) if i != 1]) # Axes over which mean/var were computed

		grad_gamma = np.sum(self.in_x_hat * grad_out_x, axis = axis, keepdims = True)
		grad_beta = np.sum(grad_out_x, axis = axis, keepdims = True)

		grad_in_x_hat = self.gamma * grad_out_x

		N_group = np.prod([grad_out_x.shape[i] for i in axis]) 

		grad_in_x_hat = self.gamma * grad_out_x
		inv_std = 1. / (np.sqrt(self.var_curr + 1e-3))
		sum_grad_in_x_hat = np.mean(grad_in_x_hat, axis = axis, keepdims = True)
		sum_grad_in_x_hat_in_x_hat = np.mean(grad_in_x_hat * self.in_x_hat, axis = axis, keepdims = True)
		grad_in_x = inv_std * (
			grad_in_x_hat
			- sum_grad_in_x_hat
			- self.in_x_hat * sum_grad_in_x_hat_in_x_hat
			)

		self.gamma -= learning_rate * grad_gamma / N_group
		self.beta -= learning_rate * grad_beta / N_group

		return grad_in_x