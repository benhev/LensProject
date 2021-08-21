import numpy as np

batch_size = 32
num_classes = 10
epochs = 10
input_shape = (28, 28, 1)
ver = 3
NAME = 'digit_recog_v{}'.format(ver)
print(NAME)

# import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from keras import backend as K
import tensorflow as tf
# from PIL import Image
from tensorflow.keras.callbacks import TensorBoard

(x_train, y_train), (x_test, y_test) = mnist.load_data()

tb = TensorBoard(log_dir='logs/{}'.format(NAME))

# print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# print(y_train.shape)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# MODEL

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adadelta',
              metrics=['accuracy'])

# model = tf.keras.models.load_model('mnist.h5')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          callbacks=tb)

model.save('models/{}'.format(NAME), overwrite=False)
print('The model has successfully trained.')
