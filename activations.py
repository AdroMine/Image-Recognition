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
	A = 1/(1+np.eZp(-Z))
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
	return(Z)
	
	
	
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
	dZ[dZ<=0] = 0
	
	return dZ

	
	