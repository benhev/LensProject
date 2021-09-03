import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from keras import backend as K
import tensorflow as tf
# from PIL import Image
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from os.path import isdir
from os import listdir
import time
import lzma
import pickle


def time_convert(seconds):
    hrs = str(np.floor(seconds / 3600).astype('int'))
    seconds = np.mod(seconds, 3600)
    mins = str(np.floor(seconds / 60).astype('int'))
    seconds = str(np.mod(seconds, 60).astype('int'))
    if len(hrs) == 1:
        hrs = '0' + hrs
    if len(mins) == 1:
        mins = '0' + mins
    if len(seconds) == 1:
        seconds = '0' + seconds
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
        X, Y = file_read(directory + "/" + file)

        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        assert len(X) == stack
        assert X.shape == Y.shape

        if len(Y.shape) > 3:
            Y = Y.reshape(shape=Y.shape[:-1])

        if ind < (num_files - 1) * stack:
            for i in range(stack):
                x[ind + i] = X[i]  # / X[i].max()
                y[ind + i] = Y[i]
            ind += len(X)

    # for i in range(len(X)):
    #     X[i] = X[i] / X[i].max()

    return (x, y), (X, Y)


batch_size = 30  # 20
# num_classes = 10
epochs = 1
input_shape = (100, 100, 1)
print('Model version: ')
ver = input()
NAME = 'Lens_CNN_v' + ver
while isdir('models/' + NAME):
    print('Model version exists. Enter different version: ')
    ver = input()
    NAME = 'Lens_CNN_v' + ver

print('Building model ' + NAME)

### Callbacks ###

tb = TensorBoard(log_dir='logs/' + NAME)  # TensorBoard
mcp = ModelCheckpoint(filepath='models/' + NAME + '/Checkpoint.h5', save_freq='epoch', verbose=1)  # Model Checkpoint
mbst = ModelCheckpoint(filepath='models/' + NAME + '/BestFit.h5', monitor='val_loss',
                       save_best_only=True, verbose=1)  # Best Model Checkpoint
estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)  # Early Stopping
redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min',
                          min_delta=1e-4)  # Adaptive Learning Rate

(x_train, y_train), (x_test, y_test) = import_data(directory='Temp Training Set', stack=100, img_shape=(100, 100))

### End Callbacks ###

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

### MODEL ###

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adadelta',
              metrics=['accuracy'])

# model = tf.keras.models.load_model('mnist.h5')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          callbacks=[tb, mcp, mbst, redlr])

# model.save('models/' + NAME)
print('The model has successfully trained.')
