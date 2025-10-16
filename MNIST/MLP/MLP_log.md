Basic architecture:
	Dense hidden layers
	Loss: cross entropy
	Train with batched SGD

Benchmark architecture:
	* asdf
	* gotta specify all hyperparams (learning rate, etc) in results report


v0.0: attempt at MLP, two hidden layers, 16 neurons each. activation: sigmoid
	- functional implementation
	- doesnt train

v1.0: linear regression, no hidden layers. activation: sigmoid
	+ data standardization
	+ with 6 epochs, takes ~8 sec. to achieve ~88% accuracy

v2.0: MLP
	+ customizable hidden layers (eg. two of sizes 256 and 128). activation: relu
	+ with 6 epochs, takes ~120 sec. to achieve ~96% accuracy

v2.1 (remastered): MLP
	+ batch vectorization
	+ other fixes (see v4.0)
		* note: no modules and no adam
	* results:
		+ blah blah !!!!!!!!
	* historical notes: 
		* was produced after v4.0.1 and helped find the bug fixed by v4.0.2

v3.0: module
	+ module implementation
	+ batch vectorization
	+ dropout (for hidden layers)
	* results:
		+ with N = 32, trains ~1.7 seconds/epoch
		+ ~96.7% accuracy by epoch 12

v3.0.1: BN
	+ batchnorm
	* results:
		+ with N = 32, trains ~2 seconds/epoch (SGD)
		+ ~96.5% accuracy by epoch 12
		+ peak ~97.7% accuracy by epoch 40

v4.0: optimizer
	+ Adam
	* results:
		+ with N = 32, trains ~3.2 seconds/epoch (old)
		+ ~98% accuracy by epoch 12
		+ peak ~98.3% accuracy by epoch 50

v4.0.1: speed optimizations
	+ dtype fix: everything cast to np.float32
	+ Adam efficiency improvement (see Kingma et al.)
	+ oh y_train before training
	* results (N = 32):
		+ SGD trains ~1.38s (old: 1.05s)/ epoch
		+ Adam trains ~2s (old: 1.65s)/ epoch (down from 2.3s [old: 2s] w/o Adam improvement)


v4.0.2: second dtype fix
	+ error: np.float32 + np.int32 -> np.float64 
	+ arises when subtracting one-hot target and softmax to find upper-most gradient
	+ fix: cast oh to dtype
	* results (N = 32):
		+ SGD trains ~ 0.7s / epoch
		+ Adam trains ~ 1.6s / epoch


	
* NOTE (8/10/2025): not sure how some of these speed benchmark numbers for v4.0 were produced. corrected with retested values
