import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from pathlib import Path
from os.path import isdir, isfile
import time
from datetime import datetime
from pickle import dump


# from os import listdir

class TimeHistory(Callback):
    def __init__(self, name, initial_epoch):
        self.name = name
        self.epoch = initial_epoch
        super().__init__()

    def on_train_begin(self, logs={}):
        # self.epoch = 0
        with open(f'models/{self.name}/model.txt', 'at') as file:
            if not self.epoch:
                file.write('\n\n\n\tEpoch\t\t\tTime\n')
                file.write('=' * 45 + '\n')
            else:
                file.write('-' * 45 + '\n')

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
        self.epoch += 1

    def on_epoch_end(self, batch, logs={}):
        with open(f'models/{self.name}/model.txt', 'at') as file:
            file.write(f'\t{self.epoch}\t\t\t{time_convert(time.time() - self.epoch_time_start)}\n')


class LensSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        with open(self.x, 'rb') as fhandle:
            _, _ = np.lib.format.read_magic(fhandle)
            shape_x, _, _ = np.lib.format.read_array_header_1_0(fhandle)

        with open(self.y, 'rb') as fhandle:
            _, _ = np.lib.format.read_magic(fhandle)
            shape_y, _, _ = np.lib.format.read_array_header_1_0(fhandle)
        assert shape_x[0] == shape_y[0]
        return int(np.ceil(shape_x[0] / self.batch_size))

    def __getitem__(self, idx):
        batch_x = read_npy_chunk(filename=self.x, start_row=idx, num_rows=self.batch_size)
        batch_y = read_npy_chunk(filename=self.y, start_row=idx, num_rows=self.batch_size)

        # batch = list(zip(batch_x, batch_y))
        # shuffle(batch)
        perm = np.random.permutation(len(batch_x))

        # batch_x = np.array([item[0] for item in batch])
        # batch_y = np.array([item[1] for item in batch])

        return batch_x[perm], batch_y[perm]


def npy_get_shape(file: str):
    with open(file, 'rb') as f:
        _, _ = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f)
    return shape


def get_file(text: str = 'Input path:'):
    user_ans = input(text)
    while not isfile(user_ans):
        user_ans = input(f'File does not exist!\n {text}')
    return user_ans


def read_npy_chunk(filename: str, start_row, num_rows):
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


def create_model(loss_func, name: str, kernel_size=(3, 3), pool_size=(2, 2), input_shape=(100, 100, 1)):
    # size_conv_layers = kwargs['size_conv_layers']
    # kernel_size = kwargs['kernel_size']
    # input_shape = kwargs['input_shape']
    # num_conv_layers = kwargs['num_conv_layers']
    # loss_func = kwargs['loss_func']
    # name = kwargs['name']

    model = Sequential(name=name)

    model.add(Conv2D(16, kernel_size=kernel_size, activation='relu', input_shape=input_shape, padding='same',
                     kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=pool_size, padding='same'))

    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=pool_size, padding='same'))

    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'))
    # model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.compile(loss=loss_func, optimizer='Adadelta', metrics=['accuracy'])

    try:
        with open(f'models/{name}/summary.txt', 'wt') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))
    except FileNotFoundError:
        print(f'Directory not found. Defaulting to current working directory at {os.getcwd()}.')
        with open(f'{name} summary.txt', 'wt') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

    return model


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


def isletter(tst: str, let: str = None):
    if len(tst) != 1:
        return False
    elif let is None:
        return tst.isalpha()
    else:
        assert len(let) == 1
        if let.islower():
            return tst == let or ord(tst) == ord(let) - 32
        else:
            return tst == let or ord(tst) == ord(let) + 32


def main():
    now = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    DATE, TIME = now.split()
    ### Model Parameters ###
    init_epoch = 0
    batch_size = int(input('Batch size: '))
    epochs = int(input('Number of epochs: '))
    # num_conv_layers = input('Number of hidden Conv2D layers: ')
    # size_conv_layers = input('Size of Conv2D layers: ')
    kernel_size = (3, 3)
    pool_size = (2, 2)
    loss_func = tf.keras.losses.mse

    # input_training = get_file(text='Training input path:')
    # label_training = get_file(text='Training label path:')
    # input_validation = get_file(text='Validation input path:')
    # label_validation = get_file(text='Validation label path:')

    input_training = 'input_validation.npy'
    label_training = 'FIXED_label_validation.npy'
    input_validation = 'input_validation.npy'
    label_validation = 'FIXED_label_validation.npy'

    # input_training = 'input_training.npy'
    # label_training = 'FIXED_label_training.npy'
    # input_validation = 'input_validation.npy'
    # label_validation = 'FIXED_label_validation.npy'

    ### validation of test and training set sizes
    shape_x = npy_get_shape(input_training)
    shape_y = npy_get_shape(label_training)
    shape_x_val = npy_get_shape(input_validation)
    shape_y_val = npy_get_shape(label_validation)

    assert shape_x == shape_y
    assert shape_x_val == shape_y_val
    NUMSAMPLE = shape_x[0]
    NUMVAL = shape_x_val[0]
    input_shape = shape_x[1:]

    ### End Model Parameters ###

    ### Identifying the model ###
    create = input('(L)oad/(C)reate model? ')
    while not (isletter(create, 'c') or isletter(create, 'l')):
        create = input('Unexpected response. l/c ?')
    create = isletter(create, 'c')

    ver = input('Model version:')
    while not ver.isdigit():
        ver = input('Improper format. Expected integer.\nModel version:')

    NAME = f'Lens_CNN_v{ver}'

    if create:
        ### Creating the model directory ###
        while isdir(f'models/{NAME}'):
            ver = input('Model version exists.\nEnter different version:')
            while not ver.isdigit():
                ver = input('Improper format. Expected integer.\nModel version:')
            NAME = f'Lens_CNN_v{ver}'
        Path(f'models/{NAME}').mkdir()
        print('Preparing log file.')
        lst = [f'Model {NAME}\nInitial training sequence performed on {DATE} at {TIME}.\n',
               f'{NUMSAMPLE} instances of {input_shape[0]}x{input_shape[1]} images with {input_shape[2]} color channels validated against {NUMVAL} test samples.\n',
               f'Batch size: {batch_size}\n', f'Number of Epochs: {epochs}\n', f'Loss Function: {loss_func.__name__}\n',
               f'Conv. kernel size: {kernel_size}\nMax and UpSampling pool size:{pool_size}\n',
               f'Training input file:{input_training}\n', f'Training label file:{label_training}\n',
               f'Validation input file:{input_validation}\n', f'Validation label file:{label_validation}\n']
        temp = []

        ### End directory creation ###

        print(f'Building model {NAME}.')
        model = create_model(name=NAME, loss_func=loss_func, kernel_size=kernel_size, input_shape=input_shape,
                             pool_size=pool_size)
    else: #Load file
        while not isdir(f'models/{NAME}'):
            ver = input('Model not found.\nEnter existing version:')
            while not ver.isdigit():
                ver = input('Improper format. Expected integer.\nModel version:')
            NAME = f'Lens_CNN_v{ver}'

        model_to_load = f'models/{NAME}/Checkpoint.h5'

        print('Preparing log file.')
        init_epoch = []
        with open(f'models/{NAME}/model.txt', 'rt') as file:
            lst = file.read().splitlines(True)
        for i in range(len(lst)):
            if 'Epoch\t\t\tTime' in lst[i]:
                ind = i - 1
            if 'Number of Epochs' in lst[i]:
                init_epoch.append(i)
        _ = 0
        for i in init_epoch:
            _ += int(lst[i].split()[-1])
        init_epoch = _

        temp = ['\n\n' + '-' * 30 + '\n', f'Additional training sequence initiated on {DATE} at {TIME}\n',
                f'{NUMSAMPLE} instances of {input_shape[0]}x{input_shape[1]} images with {input_shape[2]} color channels validated against {NUMVAL} test samples.\n',
                f'Batch size: {batch_size}\n', f'Initial Epoch: {init_epoch}\n',
                f'Number of Epochs: {epochs}\n',
                f'Training input file:{input_training}\n', f'Training label file:{label_training}\n',
                f'Validation input file:{input_validation}\n', f'Validation label file:{label_validation}\n']

        print(f'Loading model {NAME}.')
        model = load_model(model_to_load)

    ### Callbacks ###

    tb = TensorBoard(log_dir=f'logs/{NAME}')  # TensorBoard

    mcp = ModelCheckpoint(filepath=f'models/{NAME}/Checkpoint.h5', save_freq='epoch', verbose=1,
                          save_weights_only=False)  # Model Checkpoint

    mbst = ModelCheckpoint(filepath=f'models/{NAME}/BestFit_{DATE.replace("/", "-")}.h5', monitor='val_loss',
                           save_best_only=True, verbose=1, save_weights_only=False)  # Best Model Checkpoint

    estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)  # Early Stopping

    redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min',
                              min_delta=1e-4)  # Adaptive Learning Rate

    tlog = TimeHistory(name=NAME, initial_epoch=init_epoch)

    cb_names = {'tb': 'TensorBoard',
                'mcp': 'Epoch Checkpoint',
                'mbst': 'Best Checkpoint',
                'estop': 'Early Stopping',
                'redlr': 'Reduce LR on Plateau'}
    cb_dict = {'tb': tb,
               'mcp': mcp,
               'mbst': mbst,
               'estop': estop,
               'redlr': redlr}

    callbacks = [tlog]
    cb_temp = ['Epoch Timing']
    prompt = f'Callbacks to utilize. Options:\n' + '-' * 60 + '\n' + '\n'.join([f'{i[0]} -> {i[1]}' for i in
                                                                                cb_names.items()]) + '\nq -> Exit\n' + '-' * 60 + '\nWrong inputs will yield no callbacks, use the correct case.\n'

    flag = input(prompt)

    while not (isletter(flag, 'q') or len(callbacks) == len(cb_dict)):
        try:
            callbacks.append(cb_dict[flag])
            cb_temp.append(cb_names[flag])
        except KeyError:
            pass
        flag = input('Add another? (q) to exit callback selection.\n')
    temp.append('Callbacks: ' + ', '.join(cb_temp) + '\n\n')
    temp.append(input('Additional Comments:'))
    train_sequence = LensSequence(x_set=input_training, y_set=label_training,
                                  batch_size=batch_size)  # initialize a training sequence
    test_sequence = LensSequence(x_set=input_validation, y_set=label_validation,
                                 batch_size=batch_size)  # initialize a validation sequence

    ### MODEL ###
    print('Writing logs to file.')
    if create:
        lst.extend(temp)
    else:
        for line, i in zip(temp, range(ind, ind + len(temp))):
            lst.insert(i, line)
    with open(f'models/{NAME}/model.txt', 'wt') as file:
        file.writelines(lst)

    history = model.fit(train_sequence,
                        batch_size=batch_size,
                        initial_epoch=init_epoch,
                        epochs=epochs + init_epoch,
                        verbose=1,
                        validation_data=test_sequence,
                        callbacks=callbacks)
    with open(f'models/{NAME}/history_{DATE.replace("/", "-")}.json', 'xb') as file:
        dump(history.history, file)
    print(f'{NAME} has finished training sequence.')


if __name__ == '__main__':
    main()
