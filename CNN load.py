import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist

(_, _), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
x_test /= 255

model = keras.models.load_model('mnist.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

np.argmax