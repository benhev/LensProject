import numpy as np
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import Sequence
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
from random import shuffle
from pathlib import Path


def read_npy_chunk(filename, start_row, num_rows):
    """
    Reads a partial array (contiguous chunk along the first
    axis) from an NPY file.
    Parameters
    ----------
    filename : str
        Name/path of the file from which to read.
    start_row : int
        The first row of the chunk you wish to read. Must be
        less than the number of rows (elements along the first
        axis) in the file.
    num_rows : int
        The number of rows you wish to read. The total of
        `start_row + num_rows` must be less than the number of
        rows (elements along the first axis) in the file.
    Returns
    -------
    out : ndarray
        Array with `out.shape[0] == num_rows`, equivalent to
        `arr[start_row:start_row + num_rows]` if `arr` were
        the entire array (note that the entire array is never
        loaded into memory by this function).
    """
    assert start_row >= 0 and num_rows > 0
    with open(filename, 'rb') as fhandle:
        major, minor = np.lib.format.read_magic(fhandle)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fhandle)
        assert not fortran, "Fortran order arrays not supported"
        # Make sure the offsets aren't invalid.
        assert start_row < shape[0], (
            'start_row is beyond end of file'
        )
        if not start_row + num_rows <= shape[0]:
            num_rows = shape[0] - start_row
        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        row_size = np.prod(shape[1:])
        start_byte = start_row * row_size * dtype.itemsize
        fhandle.seek(start_byte, 1)
        n_items = row_size * num_rows
        flat = np.fromfile(fhandle, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])


class LensSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, num_sample):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.num_sample = num_sample

    def __len__(self):
        return int(np.ceil(self.num_sample / self.batch_size))

    def __getitem__(self, idx):
        batch_x = read_npy_chunk(filename=self.x, start_row=idx, num_rows=self.batch_size)
        batch_y = read_npy_chunk(filename=self.y, start_row=idx, num_rows=self.batch_size)

        batch = list(zip(batch_x, batch_y))
        shuffle(batch)

        batch_x = np.array([item[0] for item in batch])
        batch_y = np.array([item[1] for item in batch])

        return batch_x, batch_y


# def time_convert(seconds):
#     hrs = str(np.floor(seconds / 3600).astype('int'))
#     seconds = np.mod(seconds, 3600)
#     mins = str(np.floor(seconds / 60).astype('int'))
#     seconds = str(np.mod(seconds, 60).astype('int'))
#     if len(hrs) == 1:
#         hrs = f'0{hrs}'
#     if len(mins) == 1:
#         mins = f'0{mins}'
#     if len(seconds) == 1:
#         seconds = f'0{seconds}'
#     return ':'.join([hrs, mins, seconds])


### Model Parameters ###

NUMSAMPLE = 100000
batch_size = 20
epochs = 10
input_shape = (100, 100, 1)
num_conv_layers = 3
size_conv_layers = 16
kernel_size = (3, 3)
loss_func = tf.keras.losses.mse
input_training = 'full_input_set.npy' #'test_train_input.npy'
label_training = 'full_label_set.npy' #'test_train_label.npy'

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
    file.write(f'Number of training instances: {NUMSAMPLE}\n')
    file.write(f'Batch size: {batch_size}\n')
    file.write(f'Number of Epochs: {epochs}\n')
    file.write(f'Loss Function: {loss_func.__name__}\n')
    file.write(
        f'Model Architecture: An input layer + {num_conv_layers} hidden convolution layers with kernel_size={kernel_size} and {size_conv_layers} filters in each layer.\n\t\tEnds with a {(1, 1)} convolution layer with {1} filter.\n\n')
    print('Additional Comments')
    file.write(input())

### End directory creation ###

print(f'Building model {NAME}')

### Callbacks ###

# tb = TensorBoard(log_dir=f'logs/{NAME}')  # TensorBoard
# mcp = ModelCheckpoint(filepath=f'models/{NAME}/Checkpoint.h5', save_freq='epoch', verbose=1)  # Model Checkpoint
# mbst = ModelCheckpoint(filepath=f'models/{NAME}/BestFit.h5', monitor='val_loss',
#                        save_best_only=True, verbose=1)  # Best Model Checkpoint
# estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)  # Early Stopping
# redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min',
#                           min_delta=1e-4)  # Adaptive Learning Rate
### End Callbacks ###

# (x_train, y_train), (x_test, y_test) = import_data(directory=training_dir, stack=10000, img_shape=(100, 100))
#
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

lens_sequence = LensSequence(x_set=input_training, y_set=label_training, batch_size=batch_size, num_sample=NUMSAMPLE)

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

model.fit(lens_sequence, batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[tb])#, mcp, mbst, redlr])

print('The model has successfully trained.')
