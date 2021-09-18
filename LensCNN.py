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


class BestCheckpoint(ModelCheckpoint):
    '''
    Reimplementation of ModelCheckpoint callback with save_best_only=True to write logs to file.
    Identical to ModelCheckpoint otherwise.
    '''

    def _save_model(self, epoch, logs, batch=None):
        '''
        Internally used method.
        An extension of existing method in ModelCheckpoint.
        '''
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            with open(os.path.dirname(self.filepath) + '/model.txt', 'rt') as file:
                txt = file.readlines()
            # When editing the last line of the logs file the assumption is that TimeHistory runs before
            # BestCheckpoint as callbacks in the fitting method. For this to be true TimeHistory must ALWAYS precede
            # BestCheckpoint in the callbacks list returned from get_cbs()!
            # i.e callbacks=[...BestCheckpoint,...,TimeHistory,...] will NOT work and effectively log the previous epoch!
            # No error will be thrown over this. Beware!
            txt[-1] = txt[-1].removesuffix(
                '\n') + f'\t\t\t{str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))} with {self.monitor}={current}\n'
            with open(os.path.dirname(self.filepath) + '/model.txt', 'wt') as file:
                file.writelines(txt)
        # After saving logs to file, call the parent method to finish proper ModelCheckpoint execution.
        super()._save_model(epoch=epoch, logs=logs, batch=batch)


class TimeHistory(Callback):
    '''
    Logging of epoch timings.
    '''

    def __init__(self, name, initial_epoch):
        '''
        Constructor
        :param name: Model name.
        :param initial_epoch: Starting epoch.
        '''
        self.name = name
        self.epoch = initial_epoch
        super().__init__()

    def on_train_begin(self, logs={}):
        '''
        Internally used method.
        Commands to be performed at beginning of training.
        :param logs: Dict. Currently not in use.
        '''
        # self.epoch = 0
        with open(f'models/{self.name}/model.txt', 'at') as file:
            if not self.epoch:
                file.write('\n\n\tEpoch\t\t\tTime\t\t\tBest\n')
                file.write('=' * 85 + '\n')
            else:
                file.write('-' * 85 + '\n')

    def on_epoch_begin(self, epoch, logs={}):
        '''
        Internally used method.
        Commands to be performed at beginning of each epoch.
        :param epoch: Integer, index of epoch.
        :param logs: Dict. Currently not in use.
        '''
        self.epoch_time_start = time.time()
        self.epoch += 1

    def on_epoch_end(self, epoch, logs={}):
        '''
        Internally used method.
        Commands to be perfoemd at the end of each epoch.
        :param epoch: Integer, index of epoch.
        :param logs: Dict. Currently not in use.
        '''
        with open(f'models/{self.name}/model.txt', 'at') as file:
            file.write(f'\t{self.epoch}\t\t\t{time_convert(time.time() - self.epoch_time_start)}\n')


# Data is too large to store in memory all at once
# this sequence handles the input data in batches.
class LensSequence(tf.keras.utils.Sequence):
    '''
    Keras data sequence which reads numpy files in chunks.
    '''

    def __init__(self, x_set, y_set, batch_size):
        '''
        Constructor
        :param x_set: Directory to input as *.npy file.
        :param y_set: Directory to labels as *.npy file.
        :param batch_size: Size of batches to be supplied to model.
        '''
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        '''
        Return number of batches in sequence.
        :return: Total number of batches in LensSequence.
        '''
        with open(self.x, 'rb') as fhandle:
            _, _ = np.lib.format.read_magic(fhandle)
            shape_x, _, _ = np.lib.format.read_array_header_1_0(fhandle)

        with open(self.y, 'rb') as fhandle:
            _, _ = np.lib.format.read_magic(fhandle)
            shape_y, _, _ = np.lib.format.read_array_header_1_0(fhandle)
        assert shape_x[0] == shape_y[0]
        return int(np.ceil(shape_x[0] / self.batch_size))

    def __getitem__(self, idx):
        '''
        Get a shuffled batch.
        :param idx: Starting index.
        :return: Numpy array.
        '''
        batch_x = npy_read(filename=self.x, start_row=idx, num_rows=self.batch_size)
        batch_y = npy_read(filename=self.y, start_row=idx, num_rows=self.batch_size)

        # batch = list(zip(batch_x, batch_y))
        # shuffle(batch)
        perm = np.random.permutation(len(batch_x))

        # batch_x = np.array([item[0] for item in batch])
        # batch_y = np.array([item[1] for item in batch])

        return batch_x[perm], batch_y[perm]


def npy_get_shape(file: str):
    '''
    Return the shape of a numpy array stored in *.npy file as a numpy array.
    :param file: File directory.
    :return: Shape of stored numpy array as numpy array.
    '''
    with open(file, 'rb') as f:
        _, _ = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f)
    return shape


def get_file(text: str = 'Input path:'):
    '''
    Function to ensure input exists as a file.
    :param text: Prompt text.
    :return: Directory to existing file.
    '''
    txt = input(text)
    while not isfile(txt):
        txt = input(f'File does not exist!\n{text}')
    return txt


def npy_read(filename: str, start_row, num_rows):
    '''

    Modified from function originally written by:
    __author__ = "David Warde-Farley"
    __copyright__ = "Copyright (c) 2012 by " + __author__
    __license__ = "3-clause BSD"
    __email__ = "dwf@dwf.name"

    Reads chunks of data from a given numpy file.
    :param filename: npy file to read.
    :param start_row: Starting index.
    :param num_rows: Number of rows to read.
    :return: Numpy array.
    '''
    assert start_row >= 0 and num_rows > 0
    with open(filename, 'rb') as file:
        _, _ = np.lib.format.read_magic(file)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(file)
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
        file.seek(start_byte, 1)
        n_items = row_size * num_rows
        flat = np.fromfile(file, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])


# This function defines the architecture, change it here.
def create_model(loss_func, name: str, kernel_size=(3, 3), pool_size=(2, 2), input_shape=(100, 100, 1)):
    '''
    Creates a model according to the architecture defined inside.
    :param loss_func: Loss function to use.
    :param name: Model name.
    :param kernel_size: Convolution kernel size, (3,3) unless otherwise specified.
    :param pool_size: UpSampling and MaxPooling pool size, (2,2) unless otherwise specified.
    :param input_shape: Input image shape, (100,100,1) unless otherwise specified.
    :return: Sequetial Keras model built according to the specified architecture.
    '''
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
    '''
    Convert time from seconds to regular hh:mm:ss format.
    '''
    # This function is necessary for time logging to function
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
    '''
    Checks if tst is a single letter. If let is supplied checks whether the letters match (case insensitive).
    '''
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


def cb_menu(cb_names: dict):
    '''
    Prints callback menu from dictionary of names. Used in get_cbs()
    '''
    return f'Select callbacks to utilize. Note: time logging is on by default and cannot be turned off as it is necessary for epoch timing.\nOptions:\n' + '-' * 60 + '\n' + '\n'.join(
        [f'{i[0]} -> {i[1]}' for i in
         cb_names.items()]) + '\n\nq -> Exit\np-> Print this menu\n' + '-' * 60 + '\nInexact inputs will yield no callbacks, keys are case-sensitive.\n'


def create_log(batch_size, epochs, numsample, numval, input_training: str, input_validation: str,
               label_training: str, label_validation: str, input_shape=(100, 100, 1), kernel_size=(3, 3),
               pool_size=(2, 2), loss_func=None, first_run=False, name: str = None, init_epoch: int = None):
    '''
    Creates the log entry, whether for an existing file or for a first run
    '''
    date, tm = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')).split()
    if first_run and loss_func is not None and name is not None:
        temp = [f'Model {name}\n',
                f'Loss Function: {loss_func.__name__}\n',
                f'Conv. kernel size: {kernel_size}\nMax and UpSampling pool size:{pool_size}\n',
                '=' * 100 + '\n',
                f'Initial training sequence started on {date} at {tm}.\n']
    else:
        temp = ['\n' + '-' * 50 + '\n',
                f'Additional training sequence initiated on {date} at {tm}\n']

    temp.extend([
        f'{numsample} instances of {input_shape[0]}x{input_shape[1]} images with {input_shape[2]} color channels validated against {numval} test samples.\n',
        f'Batch size: {batch_size}\n'])
    if not first_run and init_epoch is not None:
        temp.extend([f'Initial Epoch: {init_epoch}\n'])
    elif not first_run and init_epoch is None:
        raise ValueError('Cannot define log file when first_run=True and init_epoch undefined.')

    temp.extend([f'Number of Epochs: {epochs}\n',
                 f'Training input file:{input_training}\n',
                 f'Training label file:{label_training}\n',
                 f'Validation input file:{input_validation}\n',
                 f'Validation label file:{label_validation}\n'])
    return temp


def isnat(test):
    '''
    Returns True if a number is an integer larger than 0 (Natural), and False otherwise.
    '''
    try:
        return True if int(test) > 0 else False
    except ValueError:
        return False


# Add callbacks to this function
def get_cbs(name: str, init_epoch: int = 0):
    '''
    Construct a list of callbacks to be used in model.fit().
    :param name: Model name (used to identify model in classes which save logs to file).
    :param init_epoch: Starting epoch.
    :return: List of keras callbacks.
    '''
    date, tm = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')).split()

    # 1/3 Add callbacks here
    tb = TensorBoard(log_dir=f'logs/{name}')  # TensorBoard
    mcp = ModelCheckpoint(filepath=f'models/{name}/Checkpoint.h5', save_freq='epoch', verbose=1,
                          save_weights_only=False)  # Model Checkpoint
    mbst = BestCheckpoint(filepath=f'models/{name}/BestFit_{date.replace("/", "-")}_{tm.replace(":", "")}.h5',
                          monitor='val_loss',
                          save_best_only=True, verbose=1, save_weights_only=False)  # Best Model Checkpoint
    estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)  # Early Stopping
    redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min',
                              min_delta=1e-4)  # Adaptive Learning Rate
    # 2/3 Then assign an id and add the new callback to the dictionaries
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

    # IMPORTANT NOTE:
    # TimeHistory is part of the custom logging procedure and is necessary for BestCheckpoint to function properly.
    # It MUST precede BestCheckpoint in the list of callbacks. As it is currently, BestCheckpoint is always added after TimeHistory.
    # No error will be thrown over this but it will seamlessly affect the log files, beware!
    # (See comments in BestCheckpoint._save_model() method)
    callbacks = [TimeHistory(name=name,
                             initial_epoch=init_epoch)]
    cb_temp = ['Epoch Timing']

    flag = input(cb_menu(cb_names)).lower()
    # 3/3 Finally, add an option to the menu using the assigned id from step 2
    while not (isletter(flag, 'q') or len(callbacks) - 1 == len(cb_dict)):
        try:
            if not cb_dict[flag] in callbacks:
                callbacks.append(cb_dict[flag])
                cb_temp.append(cb_names[flag])
                print(f'Added callback {cb_names[flag]}.\n{len(callbacks)}/{len(cb_dict) + 1} callbacks enabled.')
                cb_names.pop(flag)
            else:
                print('Callback exists. Nothing added.')
        except KeyError:
            if not isletter(flag, 'p'):
                print(f'Unrecognized key: {flag}, no callback added.')
            else:
                print(cb_menu(cb_names))
        if not len(callbacks) - 1 == len(cb_dict):
            flag = input('Add another? (q) to exit callback selection, (p) to print remaining callbacks.\n').lower()
        else:
            print('All callbacks enabled. Proceeding.')

    return callbacks, 'Callbacks: ' + ', '.join(cb_temp) + '\n\n'


def main():
    now = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    DATE, TIME = now.split()
    ### Model Parameters ###
    init_epoch = 0  # 0 means that the initial epoch is actually 1. This enumeration logic is pervasive in all functions which accept some form of init_epoch argument.
    batch_size = input('Batch size: ')
    while not isnat(batch_size):
        batch_size = input('Batch size should be a positive integer.\nBatch size: ')
    batch_size = int(batch_size)

    epochs = input('Number of epochs: ')
    while not isnat(epochs):
        epochs = input('Number of epochs should be a positive integer.\nNumber of epochs: ')
    epochs = int(epochs)

    kernel_size = (3, 3)
    pool_size = (2, 2)
    loss_func = tf.keras.losses.mse

    input_training = get_file(text='Training input path:')
    label_training = get_file(text='Training label path:')
    input_validation = get_file(text='Validation input path:')
    label_validation = get_file(text='Validation label path:')

    ### USED FOR TESTING. Smaller sets for faster epochs
    # input_training = 'TEST_input.npy'
    # label_training = 'TEST_label.npy'
    # input_validation = 'TEST_input.npy'
    # label_validation = 'TEST_label.npy'

    # validation of test and training set sizes
    shape_x = npy_get_shape(input_training)
    shape_y = npy_get_shape(label_training)
    shape_x_val = npy_get_shape(input_validation)
    shape_y_val = npy_get_shape(label_validation)

    assert shape_x == shape_y, "Shapes of training input and labels must match!"
    assert shape_x_val == shape_y_val, "Shapes of validation input and labels must match!"
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
    while not isnat(ver):
        ver = input('Improper format. Expected integer.\nModel version:')
    ver = int(ver)

    NAME = f'Lens_CNN_v{ver}'

    if create:
        ### Creating the model directory ###
        while isdir(f'models/{NAME}'):
            ver = input('Model version exists.\nEnter different version:')
            while not isnat(ver):
                ver = input('Improper format. Expected integer.\nModel version:')
            ver = int(ver)
            NAME = f'Lens_CNN_v{ver}'
        print('Creating model directory.')
        Path(f'models/{NAME}').mkdir()
        ### End directory creation ###

        print('Preparing log file.')
        log = create_log(batch_size=batch_size, epochs=epochs, numsample=NUMSAMPLE, numval=NUMVAL,
                         input_training=input_training, input_validation=input_validation,
                         label_training=label_training, label_validation=label_validation, input_shape=input_shape,
                         pool_size=pool_size, first_run=True, name=NAME, loss_func=loss_func)
        # temp is a list whose contents are to be appended to the log file.
        # Mostly used in the model loading logging strategy.
        temp = []

        print(f'Building model {NAME}.')
        model = create_model(name=NAME, loss_func=loss_func, kernel_size=kernel_size, input_shape=input_shape,
                             pool_size=pool_size)
    else:  # Load file
        while not isdir(f'models/{NAME}'):
            ver = input('Model not found.\nEnter existing version:')
            while not isnat(ver):
                ver = input('Improper format. Expected integer.\nModel version:')
            ver = int(ver)
            NAME = f'Lens_CNN_v{ver}'
        model_to_load = f'models/{NAME}/Checkpoint.h5'
        print('Preparing log file.')
        # init_epoch = []
        with open(f'models/{NAME}/model.txt', 'rt') as file:
            log = file.readlines()
        for i in range(len(log)):
            if 'Epoch\t\t\tTime' in log[i]:
                ind = i - 1
            if 'Number of Epochs' in log[i]:
                init_epoch += int(log[i].split()[-1])

        temp = create_log(batch_size=batch_size, epochs=epochs, numsample=NUMSAMPLE, numval=NUMVAL,
                          input_training=input_training, input_validation=input_validation,
                          label_training=label_training, label_validation=label_validation, input_shape=input_shape,
                          pool_size=pool_size, init_epoch=init_epoch)

        print(f'Loading model {NAME}.')
        model = load_model(model_to_load)

    ### Callbacks ###

    callbacks, message = get_cbs(name=NAME, init_epoch=init_epoch)

    temp.append(message)
    temp.append(input('Additional Comments:') + '\n')

    train_sequence = LensSequence(x_set=input_training, y_set=label_training,
                                  batch_size=batch_size)  # initialize a training sequence
    test_sequence = LensSequence(x_set=input_validation, y_set=label_validation,
                                 batch_size=batch_size)  # initialize a validation sequence

    ### MODEL ###
    print('Writing logs to file.')
    if create:
        log.extend(temp)
    else:
        for line, i in zip(temp, range(ind, ind + len(temp))):
            log.insert(i, line)
    with open(f'models/{NAME}/model.txt', 'wt') as file:
        file.writelines(log)
    history = model.fit(train_sequence,
                        batch_size=batch_size,
                        initial_epoch=init_epoch,
                        epochs=epochs + init_epoch,
                        verbose=1,
                        validation_data=test_sequence,
                        callbacks=callbacks)
    with open(f'models/{NAME}/history_{DATE.replace("/", "-")}_{TIME.replace(":", "")}.pickle', 'xb') as file:
        dump(history.history, file)
    print(f'{NAME} has finished training sequence.')


if __name__ == '__main__':
    main()
