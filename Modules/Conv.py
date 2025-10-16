import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time

# think about optimizing 
# sci pi sparse matrix?


class Conv2d:
	def __init__(self, C_in, C_out, kernel_size, stride = 1, padding = 'half', use_bias = True, init_method = 'He', dtype = np.float32):
		self.C_in, self.C_out = C_in, C_out
		if type(kernel_size) == int:
			self.kh, self.kw = kernel_size, kernel_size
		else:
			self.kh, self.kw = kernel_size # assume odd
		if type(stride) == int:
			self.sh, self.sw = stride, stride
		else:
			self.sh, self.sw = stride
		if padding == 'half':
			self.ph, self.pw = self.kh // 2, self.kw // 2
		elif padding == 'full':
			self.ph, self.pw = self.kh - 1, self.kw - 1
		elif padding == 'none':
			self.ph, self.pw = 0, 0
		elif type(padding) == int:
			self.ph, self.pw = padding, padding
		else:
			self.ph, self.pw = padding # expect kh > ph
		self.use_bias = use_bias
		self.dtype = dtype

		if init_method == 'He':
			self.w = (np.sqrt(2. / (self.C_in * self.kh * self.kw)) * np.random.randn(self.C_out, self.C_in, self.kh, self.kw)).astype(self.dtype)
		elif init_method == 'Xa':
			self.w = (np.sqrt(2. / ((self.C_in + self.C_out) * self.kh * self.kw)) * np.random.randn(self.C_out, self.C_in, self.kh, self.kw)).astype(self.dtype)
			# check this
		self.grad_w = np.zeros_like(self.w)

		if self.use_bias:
			self.b = np.zeros((1, C_out, 1, 1), dtype = self.dtype)
			self.grad_b = np.zeros_like(self.b)
		else:
			self.b = None
			self.grad_b = None

		self.pad_in_x = None
		self.full_win_shape = None

	def parameters(self):
		return {'params': [self.w, self.b], 'grads': [self.grad_w, self.grad_b]}

	def forward(self, in_x, **kwargs): # N, C_in, h, w
		self.in_x_shape = in_x.shape
		self.pad_in_x = np.pad(in_x, ((0, 0), (0, 0), (self.ph, self.ph), (self.pw, self.pw)), mode = 'constant') # N, C_in, h + 2ph, w + 2ph
		full_win_pad_in_x = sliding_window_view(self.pad_in_x, window_shape = (self.kh, self.kw), axis = (2, 3))
		self.full_win_shape = full_win_pad_in_x.shape # N, C_in, (h + 2ph - 2(floor kh/2)), ..., kh, kw), ie. h + 2ph - kh + 1 when kh odd

		if self.sh == 1 and self.sw == 1:
			str_win_pad_in_x = full_win_pad_in_x
		else:
			str_win_pad_in_x = full_win_pad_in_x[:, :, ::self.sh, ::self.sw, :, :] # N, C_in, reduced (h, w), kh, kw
			# ridiculously inefficient btw

		flat_str_win_pad_in_x = str_win_pad_in_x.transpose(0, 2, 3, 1, 4, 5).reshape(-1, self.C_in * self.kh * self.kw)
		flat_kernel = self.w.reshape(self.C_out, -1)

		out_x = (flat_kernel @ flat_str_win_pad_in_x.T).reshape(self.C_out, -1, *str_win_pad_in_x.shape[2:4]).transpose(1, 0, 2, 3) # N, C_out, rh, rw

		if self.use_bias:
			out_x = out_x + self.b

		return out_x

	def back(self, grad_out_x, isTraining = True, **kwargs): # N, C_out, rh, rw
		rh, rw = grad_out_x.shape[2:4]
		if self.sh == 1 and self.sw == 1:
			unstr_grad_out_x = grad_out_x
		else:
			unstr_grad_out_x = np.zeros((*grad_out_x.shape[:2], *self.full_win_shape[2:4]), dtype = self.dtype)
			unstr_grad_out_x[:, :, ::self.sh, ::self.sw] += grad_out_x # N, C_out, (rh - 1) * sh + 1, _w

		if self.kh == self.ph + 1 and self.kw == self.pw + 1: # full
			pad_unstr_grad_out_x = unstr_grad_out_x
		else:
			pad_unstr_grad_out_x = np.pad(unstr_grad_out_x, ((0, 0), (0, 0), (self.kh - self.ph - 1, self.kh - self.ph - 1), (self.kw - self.pw - 1, self.kw - self.pw - 1)), mode = 'constant')

		win_pad_unstr_grad_out_x = sliding_window_view(pad_unstr_grad_out_x, window_shape = (self.kh, self.kw), axis = (2, 3)) # N, C_out, h, w, kh, kw
		flat_win_pad_unstr_grad_out_x = win_pad_unstr_grad_out_x.transpose(0, 2, 3, 1, 4, 5).reshape(-1, self.C_out * self.kh * self.kw)


		flip_kernel = self.w[:, :, ::-1, ::-1] # C_out, C_in, kh, kw
		flat_flip_kernel = flip_kernel.transpose(1, 0, 2, 3).reshape(self.C_in, -1)
		grad_in_x = (flat_flip_kernel @ flat_win_pad_unstr_grad_out_x.T).reshape(self.C_in, -1, *win_pad_unstr_grad_out_x.shape[2:4]).transpose(1, 0, 2, 3)

		if isTraining:
			if self.sh == 1 and self.sw == 1:
				clip_pad_in_x = self.pad_in_x
				clip_unstr_grad_out_x = grad_out_x
			else:
				clip_pad_in_x = self.pad_in_x[:, :, :((rh - 1) * self.sh + self.kh), :((rw - 1) * self.sw + self.kw)]
				clip_unstr_grad_out_x = unstr_grad_out_x[:, :, :((rh - 1) * self.sh + 1), :((rw - 1) * self.sw + 1)]

			win_clip_pad_in_x = sliding_window_view(clip_pad_in_x, window_shape = ((rh - 1) * self.sh + 1, (rw - 1) * self.sw + 1), axis = (2, 3)) # N, C_in, kh, kw, (rh - 1) * sh + 1, _w
			flat_win_clip_pad_in_x = win_clip_pad_in_x.transpose(1, 2, 3, 0, 4, 5).reshape(self.C_in * self.kh * self.kw, -1)
			flat_clip_unstr_grad_out_x = clip_unstr_grad_out_x.transpose(1, 0, 2, 3).reshape(self.C_out, -1)

			N_group = np.prod(flat_clip_unstr_grad_out_x[0].shape, dtype = self.dtype) # unsure if dimensions here are right, we use 
			self.grad_w += (flat_clip_unstr_grad_out_x @ flat_win_clip_pad_in_x.T).reshape(self.C_out, self.C_in, self.kh, self.kw) / N_group

			if self.use_bias:
				self.grad_b += np.mean(grad_out_x, axis = (0, 2, 3), keepdims = True, dtype = self.dtype)

		return grad_in_x


class Conv2dTranspose: # no kernel flipping
	def __init__(self, C_in, C_out, kernel_size, stride, padding = 1, use_bias = True, init_method = 'He', dtype = np.float32):
		self.C_in, self.C_out = C_in, C_out
		if type(kernel_size) == int:
			self.kh, self.kw = kernel_size, kernel_size
		else:
			self.kh, self.kw = kernel_size # assume odd (else no center)
		if type(stride) == int:
			self.sh, self.sw = stride, stride
		else:
			self.sh, self.sw = stride
		if type(padding) == int:
			self.ph, self.pw = padding, padding
		else:
			self.ph, self.pw = padding # expect kh > ph
		self.use_bias = use_bias
		self.dtype = dtype

		if init_method == 'He':
			self.w = (np.sqrt(2. / (self.C_in * self.kh * self.kw)) * np.random.randn(self.C_out, self.C_in, self.kh, self.kw)).astype(self.dtype)
		elif init_method == 'Xa':
			self.w = (np.sqrt(2. / ((self.C_in + self.C_out) * self.kh * self.kw)) * np.random.randn(self.C_out, self.C_in, self.kh, self.kw)).astype(self.dtype)
			# check this
		self.grad_w = np.zeros_like(self.w)


	def forward(self, in_x, **kwargs): # less clean than conv
		N, _, h, w = in_x.shape

		str_in_x = np.zeros((N, C_in, h + (h - 1) * self.sh, w + (w - 1) * self.wh), dtype = self.dtype)
		str_in_x[:, :, ::self.sh, ::self.sw] += in_x
		pad_str_in_x = np.pad(str_in_x, ((0, 0), (0, 0), (self.ph, self.ph), (self.pw, self.pw)), mode = 'constant')
		win_pad_str_in_x = sliding_window_view(pad_str_in_x, window_shape = (self.kh, self.kw), axis = (2, 3))
		flat_win_pad_str_in_x = win_pad_str_in_x.transpose(0, 2, 3, 1, 4, 5).reshape(-1, self.C_in * self.kh * self.kw)

		flat_kernel = self.w.reshape(self.C_out, -1)
		out_x = (flat_kernel @ flat_win_pad_str_in_x.T).reshape(self.C_out, -1, *win_pad_str_in_x.shape[2:4]).transpose(1, 0, 2, 3)

		if self.use_bias:
			out_x = out_x + self.b

		return out_x 
		# first, put in strides
		# next, put in pad
		# then, do windows and multiply :D

	def back(self):

		# for grad_in: 
		# get rid of excess pad
		# window w/ stride s+1
		flip_kernel = self.w[:, :, ::-1, ::-1] # C_out, C_in, kh, kw
		flat_flip_kernel = flip_kernel.transpose(1, 0, 2, 3).reshape(self.C_in, -1)
		# 
		

		pass


class MaxPool2d:
	def __init__(self, shape, stride = None, dtype = np.float32):
		if type(shape) == int:
			self.ph, self.pw = shape, shape
		else:
			self.ph, self.pw = shape
		if stride is None:
			self.sh, self.sw = self.ph, self.pw 
		elif type(stride) == int:
			self.sh, self.sw = stride, stride # expect sh >= ph
		else:
			self.sh, self.sw = stride
		self.dtype = dtype
		self.mask = None
		self.h, self.w = None, None

	def parameters(self):
		return {'params': [], 'grads': []}

	def forward(self, in_x, **kwargs):
		N, C, self.h, self.w = in_x.shape
		if self.h % self.sh == 0 and self.w % self.sw == 0:
			slice_in_x = in_x.reshape(N, C, self.h // self.sh, self.sh, self.w // self.sw, self.sw).transpose(0, 1, 2, 4, 3, 5) 
		else:
			pad_in_x = np.pad(in_x, ((0, 0), (0, 0), (0, (- self.h) % self.sh), (0, (- self.w) % self.sw)), mode = 'constant')
			slice_in_x = pad_in_x.reshape(N, C, pad_in_x.shape[2] // self.sh, self.sh, pad_in_x.shape[3] // self.sw, self.sw).transpose(0, 1, 2, 4, 3, 5) # N, C, h/sh, w/sw, sh, sw

		if self.ph == self.sh and self.pw == self.sw:
			pool_in_x = slice_in_x
		else:
			pool_in_x = slice_in_x[:, :, :, :, ::self.ph, ::self.pw] # N, C, h/sh, w/sw, ph, pw

		out_x = np.max(pool_in_x, axis = (4, 5)) # N, C, h/sh, w/sw
		self.mask = (slice_in_x == out_x.reshape(*out_x.shape, 1, 1)).astype(self.dtype) # N, C, h/sh, w/sw, sh, sw
		# repeat value are unlikely

		return out_x

	def back(self, grad_out_x, **kwargs): # N, C, h/sh, w/sw
		N, C, hsh, wsw = grad_out_x.shape
		raw_grad_in_x = (self.mask * grad_out_x.reshape(*grad_out_x.shape, 1, 1)) # N, C, h/sh, w/sw, sh, sw

		if self.h % self.sh == 0 and self.w % self.sw == 0:
			return raw_grad_in_x.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, self.h, self.w)

		shape_grad_in_x = raw_grad_in_x.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, hsh * self.sh, wsw * self.sw)
		return shape_grad_in_x[:, :, ::self.h, ::self.w]

'''
class MeanPool2d:
	def __init__(self):
		pass

	def forward(self, in_x, **kwargs):
		pass

	def back(self):
		pass
'''