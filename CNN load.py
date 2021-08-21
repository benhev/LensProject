import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from PIL import Image, ImageChops
from matplotlib import pyplot as plt

ver=3
NAME='digit_recog_v{}}'.format(ver)

model = tf.keras.models.load_model('models/{}'.format(NAME))


def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)

    img = img / 255.0
    # predicting the class
    plt.matshow(img[0])
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


#
(_, _), (x_test, y_test) = mnist.load_data()
#
# # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# input_shape = (28, 28, 1)
#
# # y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
#
# # x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# # x_train /= 255
# x_test /= 255
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# np.argmax

with Image.open('8.jpg') as im:
    # plt.matshow(im)
    # im=ImageChops.invert(im)
    plt.matshow(im)
    result, acc = predict_digit(im)
    # plt.matshow(x_test[0])
    print('Result:', str(result), ' with accuracy', str(100 * acc), '.')
    plt.show()
# for i in range(10):

# i=3;
# img=x_test[i]/255.0
# print(img)
# with Image.open('8.jpg') as img:
#     print(img.convert('L'))
# plt.matshow(img)
# print('Result: ', np.argmax(model.predict(img.reshape(1, 28, 28, 1))))
# plt.show()
