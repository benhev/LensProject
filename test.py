import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import os


def process_img(im):
    im = im.convert('L')
    im = ImageChops.invert(im)
    im = im.resize((28, 28))
    im = np.array(im)
    im = tf.keras.utils.normalize(im.reshape(1, 28, 28, 1))
    # res = model.predict([img])[0]
    # return np.argmax(res), max(res)
    return im


model = tf.keras.models.load_model('models/digit_recog_v6')

for file in os.listdir('digits'):
    with Image.open('digits/' + file) as img:
        img = process_img(img)
        res = model.predict([img])[0]
        if np.argmax(res) == int(file[0]):
            test = 'Pass'
        else:
            test = 'Fail!'
        print('------------------')
        print('Filename:' + file)
        print('Prediction:', np.argmax(res), 'with accuracy:', max(res) * 100)
        print('Final Result: ', test)
