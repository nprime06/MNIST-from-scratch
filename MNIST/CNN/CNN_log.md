Basic architecture:
	Conv x 2 + max pool + conv x 2 + max pool + dense. activation: relu
	Loss: cross entropy
	Train with batched SGD
	* NOTE: specify architecture in results reports!
	* Planned: make a report of how different architecutres perform, on different LR, with and without batchnorm, etc.

v0.0: attempt at CNN
	+ data standardization
	- functional implementation
	- looped operations
	* results:
		- trains, but too slowly: with N = 32, trains ~1.2 day/epoch

v1.0: vectorized
	+ operation vectorization (with protocols such as stride_tricks)
	+ dropout in dense layer
	* results: 
		+ with N = 32, trains ~4 min/epoch (SGD)
		+ ~98.7% accuracy by epoch 6 

v1.1: fully vectorized
	+ batch vectorization
	+ with 6 epochs, takes ~20 min to achieve ~98.7% accuracy
	* results: 
		+ with N = 32, trains ~3.3 min/epoch (SGD)
		+ ~98.7% accuracy by epoch 6 

v1.1plot: has a plot
	+ plot shows accuracy on training set over time

v2.0: module and batch norm
	+ normalize kernel and bias gradients by N_group instead of N
	+ module implementation
	+ batch norm
	* see MLP_log for other details, such as: 
		+ dtype fixes
		+ oh y_train before training
	* results: 
		+ 

v2.0.1: optimizer
	+ Adam
	* see MLP_log for other fixes, such as: 
		+ Adam improvements
	* results:
		+ 

v2.1: 