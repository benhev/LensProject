import os
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from pathlib import Path
from os.path import isdir, isfile, basename
import time
from datetime import datetime
from pickle import dump
import re


class BestCheckpoint(ModelCheckpoint):
    """
    Reimplementation of ModelCheckpoint callback with save_best_only=True to write logs to file.
    Identical to ModelCheckpoint otherwise.
    """

    # There is an issue with the signature of this method, for some reason it is dynamically called with an argument
    # which is undeclared (batch=None). Through trial and error I got this to work, it might require some tinkering if
    # tensorflow/keras is updated. This may also be an issue with conflicting definitions of the keras
    # ModelCheckpoint class, once in the keras standalone package and another in the tensorflow.keras module
    def _save_model(self, epoch, logs, batch=None):
        """
        Internally used method.
        An extension of existing method in ModelCheckpoint.
        """
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            with open(os.path.dirname(self.filepath) + '/model.txt', 'rt') as file:
                txt = file.readlines()
            # When editing the last line of the logs file the assumption is that TimeHistory runs before
            # BestCheckpoint as callbacks in the fitting method. For this to be true TimeHistory must ALWAYS precede
            # BestCheckpoint in the callbacks list returned from get_cbs()!
            # callbacks=[...BestCheckpoint,...,TimeHistory,...] will NOT work and effectively log the previous epoch!
            # No error will be thrown over this. Beware!
            txt[-1] = txt[-1].removesuffix(
                '\n') + f'\t\t\t{str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))} with {self.monitor}={current}\n'
            with open(os.path.dirname(self.filepath) + '/model.txt', 'wt') as file:
                file.writelines(txt)
        # After saving logs to file, call the parent method to finish proper ModelCheckpoint execution.
        super()._save_model(epoch=epoch, logs=logs, batch=batch)


class TimeHistory(Callback):
    """
    Logging of epoch timings.
    """

    def __init__(self, name, initial_epoch):
        """
        Constructor
        :param name: Model model_dir.
        :param initial_epoch: Starting epoch.
        """
        self.name = name
        self.epoch = initial_epoch
        self.epoch_time_start = None
        super().__init__()

    def on_train_begin(self, logs={}):
        """
        Internally used method.
        Commands to be performed at beginning of training.
        :param logs: Dict. Currently not in use.
        """
        # self.epoch = 0
        with open(f'models/{self.name}/model.txt', 'at') as file:
            if not self.epoch:
                file.write('\n\n\tEpoch\t\t\tTime\t\t\tBest\n')
                file.write('=' * 85 + '\n')
            else:
                file.write('-' * 85 + '\n')

    def on_epoch_begin(self, epoch, logs={}):
        """
        Internally used method.
        Commands to be performed at beginning of each epoch.
        :param epoch: Integer, index of epoch.
        :param logs: Dict. Currently not in use.
        """
        self.epoch_time_start = time.time()
        self.epoch += 1

    def on_epoch_end(self, epoch, logs={}):
        """
        Internally used method.
        Commands to be perfoemd at the end of each epoch.
        :param epoch: Integer, index of epoch.
        :param logs: Dict. Currently not in use.
        """
        with open(f'models/{self.name}/model.txt', 'at') as file:
            file.write(f'\t{self.epoch}\t\t\t{time_convert(time.time() - self.epoch_time_start)}\n')


# Data is too large to store in memory all at once, this Sequence class handles the input data in batches.
class LensSequence(tf.keras.utils.Sequence):
    """
    Keras data sequence which reads numpy files in chunks.
    """

    def __init__(self, x_set, y_set, batch_size):
        """
        Constructor
        :param x_set: Directory to input as *.npy file.
        :param y_set: Directory to labels as *.npy file.
        :param batch_size: Size of batches to be supplied to model.
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        """
        Return number of batches in sequence.
        :return: Total number of batches in LensSequence.
        """
        with open(self.x, 'rb') as fhandle:
            _, _ = np.lib.format.read_magic(fhandle)
            shape_x, _, _ = np.lib.format.read_array_header_1_0(fhandle)

        with open(self.y, 'rb') as fhandle:
            _, _ = np.lib.format.read_magic(fhandle)
            shape_y, _, _ = np.lib.format.read_array_header_1_0(fhandle)
        assert shape_x[0] == shape_y[0]
        return int(np.ceil(shape_x[0] / self.batch_size))

    def __getitem__(self, idx):
        """
        Get a shuffled batch.
        :param idx: Starting index.
        :return: Numpy array.
        """
        batch_x = npy_read(filename=self.x, start_row=idx, num_rows=self.batch_size)
        batch_y = npy_read(filename=self.y, start_row=idx, num_rows=self.batch_size)

        # batch = list(zip(batch_x, batch_y))
        # shuffle(batch)
        perm = np.random.permutation(len(batch_x))

        # batch_x = np.array([item[0] for item in batch])
        # batch_y = np.array([item[1] for item in batch])

        return batch_x[perm], batch_y[perm]


def npy_get_shape(file: str):
    """
    Return the shape of a numpy array stored in *.npy file as a numpy array.
    :param file: File directory.
    :return: Shape of stored numpy array as numpy array.
    """
    with open(file, 'rb') as f:
        _, _ = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f)
    return shape


def get_file(text: str = 'Input path:'):
    """
    Function to ensure input exists as a file.
    :param text: Prompt text.
    :return: Directory to existing file.
    """
    txt = input(text)
    while not isfile(txt):
        txt = input(f'File does not exist!\n{text}')
    return txt


def npy_read(filename: str, start_row, num_rows):
    """

    Modified from function originally written by:
    __author__ = "David Warde-Farley"
    __copyright__ = "Copyright (c) 2012 by " + __author__
    __license__ = "3-clause BSD"
    __email__ = "dwf@dwf.model_dir"

    Reads chunks of data from a given numpy file.
    :param filename: npy file to read.
    :param start_row: Starting index.
    :param num_rows: Number of rows to read.
    :return: Numpy array.
    """
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
def create_model(loss_func, path_dir: str, kernel_size=(3, 3), pool_size=(2, 2), input_shape=(150, 150, 1)):
    """
    Creates a model according to the architecture defined inside.
    :param loss_func: Loss function to use.
    :param path_dir: Model model_dir.
    :param kernel_size: Convolution kernel size, (3,3) unless otherwise specified.
    :param pool_size: UpSampling and MaxPooling pool size, (2,2) unless otherwise specified.
    :param input_shape: Input image shape, (100,100,1) unless otherwise specified.
    :return: Sequetial Keras model built according to the specified architecture.
    """
    model = Sequential(name=basename(path_dir))

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
        with open(f'{path_dir}/summary.txt', 'wt') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))
    except FileNotFoundError:
        print(f'Directory not found. Defaulting to current working directory at {os.getcwd()}.')
        with open(f'{basename(path_dir)} summary.txt', 'wt') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

    return model


def time_convert(seconds):
    """
    Convert time from seconds to regular hh:mm:ss format.
    """
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


def isletter(tst: str, letter=None):
    """
    Checks if tst is a single letter.
    If letter is supplied checks whether the letters match (case insensitive).
    letter can be a list of letters
    """
    if len(tst) != 1:
        return False
    if isinstance(letter, list):
        for char in letter:
            if isletter(tst, char):
                return True
        return False

    elif letter is None:
        return tst.isalpha()
    else:
        assert isinstance(letter, str) and len(letter) == 1, f'"{letter}" cannot be longer than one character'
        # if letter.islower():
        #     return tst == letter or ord(tst) == ord(letter) - 32
        # else:
        #     return tst == letter or ord(tst) == ord(letter) + 32
        return tst.lower() == letter.lower()


def cb_menu(cb_names: dict):
    """
    Prints callback menu from dictionary of names. Used in get_cbs()
    """
    msg1 = f'Select callbacks to utilize. Note: time logging and model checkpoint are on by default and cannot be '
    msg2 = f'turned off as they are necessary for epoch timing and model saving (respectively).\nOptions:\n'
    msg3 = '-' * 60 + '\n' + '\n'.join([f'{i[0]} --> {i[1]}' for i in cb_names.items()])
    msg4 = '\n\nq -> Exit\np-> Print this menu\n' + '-' * 60 + '\nInexact inputs will yield no callbacks, keys are ' \
                                                               'case-sensitive.\n '
    return msg1 + msg2 + msg3 + msg4


def create_log(batch_size, epochs, numsample, numval, input_training: str, input_validation: str,
               label_training: str, label_validation: str, input_shape=(100, 100, 1), kernel_size=(3, 3),
               pool_size=(2, 2), loss_func=None, first_run=False, name: str = None, init_epoch: int = None,
               model_to_load=None):
    """
    Creates the log entry, whether for an existing file or for a first run
    """
    date, tm = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')).split()
    if first_run and loss_func is not None and name is not None:
        temp = [f'Model {name}\n',
                f'Loss Function: {loss_func.__name__}\n',
                f'Conv. kernel size: {kernel_size}\nMax and UpSampling pool size:{pool_size}\n',
                '=' * 100 + '\n',
                f'Initial training sequence started on {date} at {tm}.\n']
    else:
        if model_to_load:
            temp = ['\n' + '-' * 50 + '\n',
                    f'Additional training sequence initiated on {date} at {tm} for checkpoint {model_to_load}\n']
        else:
            raise ValueError('Must supply model_to_load for first_run=False.')
    temp.extend([
        f'{numsample} instances of {input_shape[0]}x{input_shape[1]} images with {input_shape[2]} color channels '
        f'validated against {numval} test samples.\n',
        f'Batch size: {batch_size}\n'])
    if not first_run and init_epoch is not None:
        temp.extend([f'Initial Epoch: {init_epoch}\n'])
    elif not first_run and init_epoch is None:
        raise ValueError('Cannot define log file when first_run=False and init_epoch undefined.')

    temp.extend([f'Number of Epochs: {epochs}\n',
                 f'Training input file:{input_training}\n',
                 f'Training label file:{label_training}\n',
                 f'Validation input file:{input_validation}\n',
                 f'Validation label file:{label_validation}\n'])
    return temp


def isnat(test):
    """
    Returns True if a number is an integer larger than 0 (Natural), and False otherwise.
    """
    try:
        return True if int(test) > 0 else False
    except ValueError:
        return False


# Add/change callbacks IN this function
def get_cbs(model_dir: str, init_epoch: int = 0):
    """
    Construct a list of callbacks to be used in model.fit().
    :param model_dir: Model name (used to identify model in classes which save logs to file).
    :param init_epoch: Starting epoch.
    :return: List of keras callbacks.
    """
    date, tm = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')).split()

    # 1/3 Add callbacks here
    tb = TensorBoard(log_dir=f'logs/{basename(model_dir)}')  # TensorBoard
    mbst = BestCheckpoint(filepath=f'{model_dir}/BestFit_{date.replace("/", "-")}_{tm.replace(":", "")}.h5',
                          monitor='val_loss',
                          save_best_only=True, verbose=1, save_weights_only=False)  # Best Model Checkpoint
    estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)  # Early Stopping
    redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min',
                              min_delta=1e-4)  # Adaptive Learning Rate
    # 2/3 Then assign an id and add the new callback to the dictionaries
    cb_names = {'tb': 'TensorBoard',
                'mbst': 'Best Checkpoint',
                'estop': 'Early Stopping',
                'redlr': 'Reduce LR on Plateau'}
    cb_dict = {'tb': tb,
               'mbst': mbst,
               'estop': estop,
               'redlr': redlr}

    # IMPORTANT NOTE:
    # TimeHistory is part of the custom logging procedure and is necessary for BestCheckpoint to function properly.
    # It MUST precede BestCheckpoint in the list of callbacks.
    # As it is currently, BestCheckpoint is always added after TimeHistory.
    # No error will be thrown over this but it will seamlessly affect the log files, beware!
    # (See comments in BestCheckpoint._save_model() method)
    callbacks = [TimeHistory(name=basename(model_dir), initial_epoch=init_epoch),
                 ModelCheckpoint(filepath=f'{model_dir}/Checkpoint.h5', save_freq='epoch', verbose=1,
                                 save_weights_only=False)]
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


def get_nat(name: str):
    nat = input(f'{name}: ')
    while not isnat(nat):
        nat = input(f'{name} should be a positive integer.\n{name}: ')
    return int(nat)


def validate_data(training: tuple, validation: tuple):
    tr_inp_shape, tr_lab_shape = list(map(npy_get_shape, training))
    val_inp_shape, val_lab_shape = list(map(npy_get_shape, validation))

    assert tr_inp_shape == tr_lab_shape, 'Shapes of training input and labels must match!'
    assert val_inp_shape == val_lab_shape, 'Shapes of validation input and labels must match!'

    return [tr_inp_shape[0], val_inp_shape[0], tr_inp_shape[1:]]


def get_dir(target, new: bool, base=''):
    if base:
        assert isdir(base), f'{base} must be an existing directory.'
        base = base.replace('\\', '/').removesuffix('/')
    path_dir = (base + '/' if base else '') + input(f'Input {target} directory: {base + "/" if base else ""}')
    while (isdir(path_dir) and new) or (not isdir(path_dir) and not new):
        prompt = f'Directory {path_dir} exists!' if new else f'Directory {path_dir} does not exist!'
        print(prompt)
        path_dir = (base + '/' if base else '') + input(f'Input {target} directory: {base + "/" if base else ""}')

    if new:
        Path(path_dir).mkdir()

    return path_dir


def opts_menu(txt: str, resp_dic: dict = {}):
    options = re.findall(r'[(\[{](\w|\d+)[)\]}]', txt) or None
    if options is None:
        raise ValueError('No bracketed characters/numbers in input string.')
    else:
        options = sorted([x.lower() for x in options])
        if resp_dic:
            assert sorted(resp_dic.keys()) == options, 'Response keys do not match text.'

    response = input(txt + '\n')
    while not isletter(response, options):
        response = input(f'Unexpected response.Options:\n{", ".join(options)}\n')

    return resp_dic.get(response, response)


def dic_menu(dic: dict, init=''):
    if len(dic) == 1:
        return list(dic.values())[0]
    prompt = init + f'\n' + '-' * 60 + '\n' + '\n'.join(
        [f'{i} --> {j}' for i, j in dic.items()]) + '\n' + '-' * 60 + '\n'
    dic = dict(zip([str(x) for x in dic.keys()], dic.values()))

    result = dic.get(input(prompt), None)
    while result is None:
        print('Unexpected response.')
        result = dic.get(input(prompt), None)
    return result


def initiate_training():
    batch_size = get_nat('Batch size')
    epochs = get_nat('Number of epochs')
    # input_training = get_file(text='Training input path:')
    # label_training = get_file(text='Training label path:')
    # input_validation = get_file(text='Validation input path:')
    # label_validation = get_file(text='Validation label path:')

    ### USED FOR TESTING. Smaller sets for faster epochs
    input_training = 'TEST_input.npy'
    label_training = 'TEST_label.npy'
    input_validation = 'TEST_input.npy'
    label_validation = 'TEST_label.npy'

    return [batch_size, epochs, (input_training, label_training), (input_validation, label_validation)]


def create_cnn(loss_func=tf.keras.losses.mse, kernel_size=(3, 3), pool_size=(2, 2)):
    batch_size, epochs, training, validation = initiate_training()
    num_sample, num_val, input_shape = validate_data(training=training, validation=validation)
    # input_training, label_training = training
    # input_validation, label_validation = validation

    ### Creating the model directory ###
    model_dir, model_name = (lambda x: (x, basename(x)))(get_dir(target='model', new=True, base='models/'))
    print('Creating model directory.')
    callbacks, callback_log = get_cbs(model_dir=model_dir, init_epoch=0)
    print('Preparing log file.')
    log = create_log(batch_size=batch_size, epochs=epochs, numsample=num_sample, numval=num_val,
                     input_training=training[0], input_validation=validation[0],
                     label_training=training[1], label_validation=validation[1], input_shape=input_shape,
                     pool_size=pool_size, kernel_size=kernel_size, first_run=True, name=model_name,
                     loss_func=loss_func)
    # temp is a list whose contents are to be appended to the log file.
    # Mostly used in the model loading logging strategy.
    log.append(callback_log)
    log.append(input('Additional Comments:') + '\n')
    write_log(model_dir=model_dir, log=log)
    print(f'Building model {model_name}.')
    model = create_model(loss_func=loss_func, path_dir=model_dir, kernel_size=kernel_size, pool_size=pool_size,
                         input_shape=input_shape)

    train_model(model=model, batch_size=batch_size, epochs=epochs, training=training, validation=validation,
                callbacks=callbacks, model_dir=model_dir)


def load_cnn(**kwargs):
    batch_size, epochs, training, validation = initiate_training()
    num_sample, num_val, input_shape = validate_data(training=training, validation=validation)
    model_dir, model_name = (lambda x: (x, basename(x)))(get_dir(target='model', new=False, base='models/'))
    model_options = (lambda y: dict(zip(range(len(y)), y)))([basename(x) for x in glob.glob(f'{model_dir}/*.h5')])
    if model_options:
        model_to_load = dic_menu(dic=model_options, init='Choose checkpoint to load')
    else:
        raise FileNotFoundError('No *.h5 files found in model directory.')
    print('Preparing log file.')
    # init_epoch = []
    log, ind, init_epoch = fetch_log(model_dir)
    callbacks, callback_log = get_cbs(model_dir=model_dir, init_epoch=init_epoch)
    temp = create_log(batch_size=batch_size, epochs=epochs, numsample=num_sample, numval=num_val,
                      input_training=training[0], input_validation=validation[0],
                      label_training=training[1], label_validation=validation[1],
                      input_shape=input_shape, init_epoch=init_epoch, model_to_load=model_to_load)
    temp.append(callback_log)
    temp.append(input('Additional Comments:') + '\n')
    write_log(model_dir=model_dir, log=log, insert=temp, insert_pos=ind)

    print(f'Loading {model_to_load} checkpoint of model {model_name}.')
    model = load_model(f'{model_dir}/{model_to_load}')

    train_model(model=model, batch_size=batch_size, epochs=epochs, training=training, validation=validation,
                callbacks=callbacks, model_dir=model_dir, init_epoch=init_epoch)


def write_log(model_dir, log, insert=[], insert_pos=None):
    if insert and insert_pos:
        for line, i in zip(insert, range(insert_pos, insert_pos + len(insert))):
            log.insert(i, line)
    elif bool(insert) ^ bool(insert_pos):  # (insert and not insert_pos) or (not insert and insert_pos):
        raise ValueError('insert text and insert_pos must be supplied to insert text.')
    with open(f'{model_dir}/model.txt', 'wt') as file:
        print('Writing logs to file.')
        file.writelines(log)


def fetch_log(model_dir):
    init_epoch = 0
    with open(f'{model_dir}/model.txt', 'rt') as file:
        log = file.readlines()
    for i in range(len(log)):
        if 'Epoch\t\t\tTime' in log[i]:
            ind = i - 1
        if 'Number of Epochs' in log[i]:
            init_epoch += int(log[i].split()[-1])

    return [log, ind, init_epoch]


def train_model(model, batch_size, epochs, training, validation, callbacks, model_dir, init_epoch=0):
    now = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    date, tm = now.split()

    input_training, label_training = training
    input_validation, label_validation = validation
    train_sequence = LensSequence(x_set=input_training, y_set=label_training,
                                  batch_size=batch_size)  # initialize a training sequence
    test_sequence = LensSequence(x_set=input_validation, y_set=label_validation,
                                 batch_size=batch_size)  # initialize a validation sequence
    history = model.fit(train_sequence,
                        batch_size=batch_size,
                        initial_epoch=init_epoch,
                        epochs=epochs + init_epoch,
                        verbose=1,
                        validation_data=test_sequence,
                        callbacks=callbacks)
    with open(f'{model_dir}/history_{date.replace("/", "-")}_{tm.replace(":", "")}.pickle', 'xb') as file:
        dump(history.history, file)
    print(f'{basename(model_dir)} has finished training sequence.')


def main():
    opts_menu('(L)oad/(C)reate model? ', {'c': create_cnn, 'l': load_cnn})()


if __name__ == '__main__':
    main()
