import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential

model = keras.models.load_model('mnist.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
