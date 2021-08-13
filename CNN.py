import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K

(x_train, y_train) , (x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)
