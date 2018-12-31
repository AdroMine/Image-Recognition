import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split # for train-test partition

# read images of pikachu and other pokemon

# first read pikachu images [186]
path = "d:/PData/image-recognition/images/"    
filelist = os.listdir(path+'positive/')
pikachu_orig = np.array([np.array(Image.open(path+'positive/'+fname)) for fname in filelist])

# now read non-pikachu images [150]
filelist = os.listdir(path+'negative/')
pokemon_orig = np.array([np.array(Image.open(path+'negative/'+fname)) for fname in filelist])

# now reshape them
pikachu = pikachu_orig.reshape(pikachu_orig.shape[0],-1)
pokemon = pokemon_orig.reshape(pokemon_orig.shape[0],-1)

# now
fullset = np.concatenate((pikachu,pokemon),axis = 0)
fullset = fullset/255

# create label 1 for Pikachu, 0 for non-pikachu
labels = np.concatenate((np.ones(pikachu.shape[0],int),np.zeros(pokemon.shape[0],int)))
labels = labels.reshape(fullset.shape[0],1)

# partition data into training, validation and test set
x_train,x_test,y_train,y_test = train_test_split(fullset,labels,test_size = 0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

x_train,x_val,x_test,y_train,y_val,y_test = x_train.T,x_val.T,x_test.T,y_train.T,y_val.T,y_test.T

