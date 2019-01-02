import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split # for train-test partition
import matplotlib.pyplot as plt
from IPython.display import display
from dnn_functions import *

# read images of pikachu and other pokemon

# first read pikachu images [186]
path = "d:/PData/image-recognition/images/"    
filelist = os.listdir(path+'positive/')
pikachu_orig = np.array([np.array(Image.open(path+'positive/'+fname)) for fname in filelist])

# now read non-pikachu images [150]
filelist = os.listdir(path+'negative/')
pokemon_orig = np.array([np.array(Image.open(path+'negative/'+fname)) for fname in filelist])

plt.figure(figsize=(10,10))

for i,image in enumerate([10,20]):
    plt.subplot(2,2,i+1)
    plt.imshow(pikachu_orig[np.random.randint(110,size=1)[0]])
for i,image in enumerate([10,20]):
    plt.subplot(2,2,i+3)
    plt.imshow(pokemon_orig[np.random.randint(110,size=1)[0]])


# Unpacking each image into a vector
pikachu = pikachu_orig.reshape(pikachu_orig.shape[0],-1)
pokemon = pokemon_orig.reshape(pokemon_orig.shape[0],-1)

print("Dimension of Pikachu matrix are: " + str(pikachu.shape))
print("Dimension of Pokemon matrix are: " + str(pokemon.shape))

fullset = np.concatenate((pikachu,pokemon),axis = 0)
fullset = fullset/255

# create label 1 for Pikachu, 0 for non-pikachu
labels = np.concatenate((np.ones(pikachu.shape[0],int),np.zeros(pokemon.shape[0],int)))
labels = labels.reshape(fullset.shape[0],1)

# partition data into training, validation and test set
x_train,x_test,y_train,y_test = train_test_split(fullset,labels,test_size = 0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

x_train,x_val,x_test,y_train,y_val,y_test = x_train.T,x_val.T,x_test.T,y_train.T,y_val.T,y_test.T

print("Dimensions of Training set are:" + str(x_train.shape))
print("No of pikachu images in training set:" + str(np.sum(y_train)))

print("\n")

print("Dimensions of Validation set are:" + str(x_val.shape))
print("No of pikachu images in validation set:" + str(np.sum(y_val)))


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False): 
    """
    Implements a L-layer neural network with the following arch: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, 64 * 64 * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    #np.random.seed(1)
    costs = []                         # store costs to plot cost with iterations
    
    # Parameters initialization
    parameters = initialise_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

# Layer parameters
layers_dims = [12288, 20, 7, 5, 1] 

parameters =L_layer_model(x_train, y_train, layers_dims,learning_rate = 0.0075,num_iterations = 2500, print_cost = True)

pred_train = predict(x_train, y_train, parameters)
pred_val = predict(x_val, y_val, parameters)                

