import numpy as np

def sigmoid(Z):
	"""
	Compute the sigmoid of Z
	
	Argument:
	Z - a scalar or numpy array
	
	Returns:
	A - sigmoid of Z
	cache - returns original Z as well which will come in handy during backward propagation
	"""
	A = 1/(1+np.exp(-Z))
	cache = Z
	return A,cache

def sigmoid_backwards(dA,cache):
	"""
	Implement gradient descent for single sigmoid unit
	
	Arguments:
	dA - post-activation gradient (any shape)
	cache - from forward prop (Z)
	
	Returns:
	dZ - gradient of the cost with respect to Z (dJ/dZ)
	"""
	
	Z = cache
	s = 1/(1+np.exp(-Z))
	dZ = dA * s * (1-s)
	return(dZ)
	
	
	
def relu(Z):
	"""
	Compute the Rectified Linear Unit (ReLU) of Z
	
	Arguments :
	Z - a scalar or numpy array
	
	Returns:
	A - ReLU of Z (same shape)
	cache - to use during backward prop
	"""
	A = np.maximum(0,Z)
	cache = Z
	return A,cache
	
def relu_backward(dA,cache):
	"""
	Implement gradient descent for single ReLU unit
	
	Arguments:
	dA - post-activation gradient (any shape)
	cache - from forward prop (Z)
	
	Returns:
	dZ - gradient of the cost with respect to Z (dJ/dZ)
	"""
	
	Z = cache
	dZ = np.array(dA,copy = True)
	dZ[Z <= 0] = 0
	
	#assert(dZ.shape == Z.shape)
	return dZ

def initialise_parameters_deep(layer_dims):
	"""
	Arguments:
	layer_dims - python array (list) containing the dimensions of each layer of DNN
	
	Returns:
	parameters - python dictionary containing parameters "W1","b1" ... "WL","bL"
		where Wi - is weight matrix of shape(layer_dims[l],layer_dims[l-1]) 
		and bi - is bias vector of shape (layer_dims[l],1)
	"""
	
	parameters = {}
	L = len(layer_dims)
	
	for l in range(1,L):
		parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) # or could also use * 0.01
		parameters['b'+str(l)] = np.zeros( ((layer_dims[l],1)) )
	
	return parameters

def linear_forward(A,W,b):
	"""
	Implement the linear part of a single layer's forward propagation step
	
	Arguments:
	A - activation matrix from previous layer (X if 1st layer) (n[l-1],m)
	W - weight matrix size(n[l],n[l-1])
	b - bias vector size(n[l],1)
	
	Returns:
	Z - pre-activation parameter
	cache - python dict with A,W,b for backward pass
	"""
	
	Z = W.dot(A) + b
	cache = (A,W,b)
	return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
	"""
	Implement forward pass for one layer (linear->activation (sigmoid/relu/etc.)
	
	Arguments:
	A_prev - activations from previous layer (or X): (n[l-1],m)
	W - weight matrix: (n[l],n[l-1])
	b - bias vector: (n[l],1)
	activation - string that defines whether to use sigmoid or ReLU for activation	
	"""
	
	Z,linear_cache = linear_forward(A_prev,W,b)
	if activation == "sigmoid":
		A,activation_cache = sigmoid(Z)
	
	elif activation == "relu":
		A,activation_cache = relu(Z)
	
	cache = (linear_cache,activation_cache)
	
	return A,cache
	
def L_model_forward(X,parameters):
	"""
	Implement one complete forward pass for Neural Network where structure is
	[Linear->ReLU]*(L-1)  --> Linear->Sigmoid 
	
	Arguments:
	X - input data
	parameters - output of initialise_parameters_deep() created above
	
	Returns:
	AL - final post-activation value (y^hat)
	caches - list of caches containing:
				every cache of relu forwards (L-1 of them, indexed from 0 to L-2)
				cache of last sigmoid pass (just one, indexed L-1)
	"""
	
	caches = []
	A = X
	L = len(parameters) // 2  # no of layers in DNN
	
	# Implementing first L-1 layers of relu activations
	# store cache from each step
	for l in range(1,L):
		A_prev = A
		A,cache = linear_activation_forward(A_prev,parameters['W'+str(l)], parameters['b'+str(l)],activation = "relu")
		caches.append(cache)
	
	# Implement last layer with sigmoid activation
	# store last cache
	AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation = "sigmoid")
	caches.append(cache)
	
	return AL,caches

def compute_cost(AL,Y):
	"""
	Compute cost after one forward pass of DNN
	
	Arguments:
	AL - probability vector corresponding to label predictions: (1,m)
	Y - true 'label' vector (1,m)
	
	Returns:
	cost -- cross-entropy cost
	"""
	
	m = Y.shape[1]
	
	cost = (1/m)* (-np.dot(Y,np.log(AL).T) - np.dot(1-Y,np.log(1-AL).T))
	
	cost = np.squeeze(cost)
	
	return cost
	
def linear_backward(dZ,cache):
	"""
	Implement the linear part of backward pass for one layer
	
	Arguments:
	dZ - dJ/dZ for current layer
	cache - (A_prev,W,b) from the forward pass for the current layer
	
	Returns:
	dA_prev - gradient of cost for A_prev (l-1 layer)
	dW - dJ/dW for current layer
	db - dJ/db for current layer
	"""
	A_prev,W,b = cache
	m = A_prev.shape[1]
	
	dW = 1./m * np.dot(dZ,A_prev.T)
	db = 1./m * np.sum(dZ,axis = 1,keepdims = True)
	dA_prev = np.dot(W.T,dZ)
	
	return dA_prev,dW,db
	
def linear_activation_backward(dA,cache,activation):
	"""
	Implement backward pass for linear->activation layer
	
	Arguments:
	dA - post-activation gradient for current layer
	cache - (linear_cache,activation_cache) from forward pass
	activation - relu/sigmoid string
	
	Returns:
	dA_prev - dJ/dA_prev (previous layer)
	dW - dJ/dW (same layer)
	db - dJ/db (same layer)
	"""
	
	linear_cache,activation_cache = cache
	
	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backwards(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)
	
	return dA_prev,dW,db
	
def L_model_backward(AL,Y,caches):
	"""
	Implement one back pass (Linear->ReLU L-1 times -> Linear->Sigmoid) to generate gradients
	
	Arguments:
	AL - output of forward pass (L_model_forward())
	Y - true "label" vector
	caches -- cache from linear_activation_forward() for each layer
	
	Returns:
	grads - dictionary with the gradients
			grads[dAl,dWl,dbl]
	"""
	
	grads = {}
	L = len(caches) # no of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # just safety
	
	# Initialise backpass with dAL
	dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
	
	# Lth layer (Sigmoid -> Linear) gradients
	current_cache = caches[L-1] # caches from 0 to L-1
	grads["dA" + str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
	
	for l in reversed(range(L-1)):
		# l th layer ReLU -> Linear gradients
		current_cache = caches[l] # starts with L-2
		dA_prev_temp, dW_temp,db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
		grads["dA"+str(l)] = dA_prev_temp
		grads["dW"+str(l+1)] = dW_temp
		grads["db"+str(l+1)] = db_temp
	
	return grads
	
def update_parameters(parameters,grads,learning_rate):
	"""
	Update parameters using gradient descent
	
	Arguments:
	parameters - dictionary with parameters
	grads - dict with gradients computed from back pass
	
	Returns:
	parameters - new updated parameters W & b for each layer
	"""
	
	L = len(parameters) // 2
	
	# update each parameter
	for l in range(L):
		parameters["W"+str(l+1)] -= learning_rate*grads["dW"+str(l+1)]
		parameters["b"+str(l+1)] -= learning_rate*grads["db"+str(l+1)]
	
	return parameters
	
def predict(X,y,parameters):
	"""
	Predict using developed DNN
	
	Arguments:
	X - Input dataset (multiple inputs) to make prediction for
	y - true labels for testing accuracy
	parameters - parameters of trained model
	
	Returns:
	p - prediction for X	
	"""
	
	m = X.shape[1]
	n = len(parameters) // 2 # no of layers in NN
	p = np.zeros((1,m))
	
	# use forward pass to compute
	probs,caches = L_model_forward(X,parameters)
	
	# convert probabilities to 0-1 using threshold of 0.5
	for i in range(0,probs.shape[1]):
		if probs[0,i] > 0.5:
			p[0,i] = 1
		else:
			p[0,i] = 0
	
	print("Accuracy:" + str(round(np.sum((p==y)/m)),2))
	
	return p

