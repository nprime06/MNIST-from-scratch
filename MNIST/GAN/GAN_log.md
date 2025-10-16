Basic GAN architecture: 
	Dense layers
		Gen: batch norm + relu (tanh output). Latent(100) -> ... -> 784 (image)
		Dis: leaky relu + dropout (sigmoid output). 784 (image) -> ... -> 1 (classification)
	Loss: modified minimax cross entropy
		Gen: L = - log(D(G(z))
		Dis: L = - (log(D(x)) + log(1 - D(G(z))))
	Train with batched SGD

v0: attempt at GAN
	- functional implementation
	+ batch vectorized
	* results: learns some features (such as setting outer pixels dark) but very poorly 
		- Generated images suck (incoherent noise at the cetner) even after extensively playing with architecture
		- Without reg techniques even some brief learning rate tuning didn't help

v1: GAN
	+ module implementation
	+ batch norm 
	+ dropout
	* results: 
		* with N = 32, BN, Dropout, layers 256, 512; 512, 256 and D/G training once per epoch, ~15 sec per epoch 
		+ By tuning learning rate (see sample 6, uses high lr) it learns pretty well!

v2: optimizer
	+ Adam
	* results:
		* same architecture as before, ~23 sec per epoch 
		+ Clear convergence within 10 epochs, learns decently
		- Slight mode collapse - doesn't produce 2 often, for example

v2: speed optimizations
	+ everything cast to np.float32
	+ Adam efficiency improvement (see Kingma et al.)
	+ results (N = 32, D/G once per epoch):
		+ SGD trains ~7.5s / epoch
		+ Adam trains ~10.25s / epoch (down from 14s w/o Adam improvement)