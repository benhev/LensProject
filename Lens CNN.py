import numpy as np
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from keras import backend as K
import tensorflow as tf
# from PIL import Image
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from os.path import isdir
from os import listdir
# import time
import lzma
import pickle
from pathlib import Path


def time_convert(seconds):
    hrs = str(np.floor(seconds / 3600).astype('int'))
    seconds = np.mod(seconds, 3600)
    mins = str(np.floor(seconds / 60).astype('int'))
    seconds = str(np.mod(seconds, 60).astype('int'))
    if len(hrs) == 1:
        hrs = f'0{hrs}'
    if len(mins) == 1:
        mins = f'0{mins}'
    if len(seconds) == 1:
        seconds = f'0{seconds}'
    return ':'.join([hrs, mins, seconds])


def file_read(directory):
    with lzma.open(directory, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def import_data(directory: str, stack=10000, img_shape=(100, 100)):
    num_files = len(listdir(directory))
    x, y = np.zeros(shape=((num_files - 1) * stack, img_shape[0], img_shape[1], 1)), np.zeros(
        shape=((num_files - 1) * stack, img_shape[0], img_shape[1]))
    ind = 0
    for file in listdir(directory):
        X, Y = file_read(f'{directory}/{file}')

        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        assert len(X) == stack
        assert X.shape == Y.shape

        if len(Y.shape) > 3:
            Y = Y.reshape(shape=Y.shape[:-1])
            np.ndarray.reshape()

        if ind < (num_files - 1) * stack:
            for i in range(stack):
                x[ind + i] = X[i]  # / X[i].max()
                y[ind + i] = Y[i]
            ind += len(X)

    # for i in range(len(X)):
    #     X[i] = X[i] / X[i].max()

    return (x, y), (X, Y)


### Model Parameters ###

batch_size = 500  # 20
epochs = 1
input_shape = (100, 100, 1)
num_conv_layers = 1
size_conv_layers = 64
kernel_size = (3, 3)
loss_func = tf.keras.losses.mse
training_dir = 'Temp Training Set'

### End Model Parameters ###

### Naming the model and creating the model directory ###

print('Model version:')
ver = input()
NAME = f'Lens_CNN_v{ver}'
while isdir(f'models/{NAME}'):
    print('Model version exists. Enter different version:')
    ver = input()
    NAME = f'Lens_CNN_v{ver}'
Path(f'models/{NAME}').mkdir()
print('Creating log file.')

with open(f'models/{NAME}/model.txt', 'wt') as file:
    file.write(f'Model {NAME}\n')
    file.write(f'Batch size: {batch_size}\n')
    file.write(f'Number of Epochs: {epochs}\n')
    file.write(f'Loss Function: {loss_func.__name__}\n')
    file.write(
        f'Model Architecture: An input layer + {num_conv_layers} hidden convolution layers with kernel_size={kernel_size} and {size_conv_layers} filters in each layer.\n\t Ends with a {(1, 1)} convolution layer with {1} filter.\n\n')
    print('Additional Comments')
    file.write(input())

### End directory creation ###

print(f'Building model {NAME}')

### Callbacks ###

tb = TensorBoard(log_dir=f'logs/{NAME}')  # TensorBoard
mcp = ModelCheckpoint(filepath=f'models/{NAME}/Checkpoint.h5', save_freq='epoch', verbose=1)  # Model Checkpoint
mbst = ModelCheckpoint(filepath=f'models/{NAME}/BestFit.h5', monitor='val_loss',
                       save_best_only=True, verbose=1)  # Best Model Checkpoint
estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)  # Early Stopping
redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min',
                          min_delta=1e-4)  # Adaptive Learning Rate
### End Callbacks ###

(x_train, y_train), (x_test, y_test) = import_data(directory=training_dir, stack=10000, img_shape=(100, 100))

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

### MODEL ###

model = Sequential()

model.add(Conv2D(size_conv_layers, kernel_size=kernel_size, activation='relu', input_shape=input_shape, padding='same'))
# model.add(MaxPool2D(pool_size=(2, 2),padding='same'))
for i in range(num_conv_layers):
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2),padding='same'))

# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

model.add(Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same'))
model.compile(loss=loss_func, optimizer='Adadelta',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          callbacks=[tb, mcp, mbst, redlr])

print('The model has successfully trained.')
