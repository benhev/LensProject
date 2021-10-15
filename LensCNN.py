import os
import glob

import numpy as np
from tensorflow.keras.utils import Sequence as Keras_Sequence
from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from pathlib import Path
from os.path import isdir, isfile, basename, dirname
import time
from datetime import datetime
from pickle import dump
from re import findall, compile as rcompile, match as rmatch, escape


class MyModelCheckpoint(ModelCheckpoint):
    def _save_model(self, epoch, logs=None, batch=None):
        regex = rcompile(
            escape(dirname(self.filepath)) + r'\\[cC]heckpoint[-_\w]*epoch=\d{3}[-_\s]*val_loss=\d+.\d{4}\.h5')
        prev_epoch = [f for f in glob.glob(f'{dirname(self.filepath)}/*.h5') if
                      regex.match(f) and f'epoch={epoch:03d}' in f]
        # epoch is in fact 1 less than the number of epoch running
        assert len(prev_epoch) <= 1, 'Found more than one previous epoch.'
        prev_epoch = prev_epoch[0] if len(prev_epoch) == 1 else None
        if prev_epoch:
            os.remove(prev_epoch)
        super()._save_model(epoch=epoch, logs=logs, batch=batch)


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
        log_dir = os.path.dirname(self.filepath).removesuffix('/BestFit') + '/model.txt'
        if self.monitor_op(current, self.best):
            with open(log_dir, 'rt') as file:
                txt = file.readlines()
            # When editing the last line of the logs file the assumption is that TimeHistory runs before
            # BestCheckpoint as callbacks in the fitting method. For this to be true TimeHistory must ALWAYS precede
            # BestCheckpoint in the callbacks list returned from get_cbs()!
            # callbacks=[...BestCheckpoint,...,TimeHistory,...] will NOT work and effectively log the previous epoch!
            # No error will be thrown over this. Beware!
            txt[-1] = txt[-1].removesuffix(
                '\n') + f'\t{str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))} with {self.monitor}={current}\n'
            with open(log_dir, 'wt') as file:
                file.writelines(txt)
        # After saving logs to file, call the parent method to finish proper ModelCheckpoint execution.
        # the function's signature has no batch argument but in runtime it receives it as a call
        # I've left this as it is because it works
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

    def on_train_begin(self, logs=None):
        """
        Internally used method.
        Commands to be performed at beginning of training.
        :param logs: Dict. Currently not in use.
        """
        with open(f'models/{self.name}/model.txt', 'at') as file:
            if not self.epoch:
                file.write('\n\n\tEpoch\t\t\tTime\t\t\tBest\n')
                file.write('=' * 100 + '\n')
            else:
                file.write('-' * 100 + '\n')

    def on_epoch_begin(self, epoch, logs=None):
        """
        Internally used method.
        Commands to be performed at beginning of each epoch.
        :param epoch: Integer, index of epoch.
        :param logs: Dict. Currently not in use.
        """
        self.epoch_time_start = time.time()
        self.epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        Internally used method.
        Commands to be perfoemd at the end of each epoch.
        :param epoch: Integer, index of epoch.
        :param logs: Dict. Currently not in use.
        """
        with open(f'models/{self.name}/model.txt', 'at') as file:
            file.write(f'\t{self.epoch}\t\t\t{time_convert(time.time() - self.epoch_time_start)}\n')


# Data is too large to store in memory all at once, this Sequence class handles the input data in batches.
class LensSequence(Keras_Sequence):
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
    return sanitize_path(txt)


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
def create_model(loss_func, optimizer, metric, path_dir: str, input_shape: tuple, kernel_size=(3, 3), pool_size=(2, 2),
                 ):
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
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metric)

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
               pool_size=(2, 2), loss_func=None, optimizer=None, metric=None, first_run=False, name: str = None,
               init_epoch: int = None,
               model_to_load=None):
    """
    Creates the log entry, whether for an existing file or for a first run
    """
    date, tm = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')).split()
    if first_run and all([loss_func, name, optimizer, metric]):
        temp = [f'Model {name}\n',
                f'Loss Function: {loss_func if isinstance(loss_func, str) else loss_func.__name__}\n',
                f'Optimizer: {optimizer if isinstance(optimizer, str) else optimizer._name}\n',
                f'Metrics: ' + ', '.join([x if isinstance(x, str) else x.name for x in metric])+'\n',
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


# TODO: Redo this function so that it creates callbacks when needed and requests the appropriate parameters

# Add/change callbacks IN this function
def get_cbs(model_dir: str, init_epoch: int = 0, auto=False):
    """
    Construct a list of callbacks to be used in model.fit().
    :param model_dir: Model name (used to identify model in classes which save logs to file).
    :param init_epoch: Starting epoch.
    :param auto: False by default. Providing a list of keys will append the respective callbacks.
    :return: List of keras callbacks.
    """
    date, tm = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')).split()

    # 1/3 Add callbacks here
    tb = TensorBoard(log_dir=f'logs/{basename(model_dir)}')  # TensorBoard
    mbst = BestCheckpoint(
        filepath=f'{model_dir}/BestFit/BestFit--{date.replace("/", "-")}_{tm.replace(":", "")}--' + 'epoch={epoch:03d}--val_loss={val_loss:.4f}.h5',
        monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=False)  # Best Model Checkpoint
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
                 # ModelCheckpoint(filepath=f'{model_dir}/Checkpoint.h5', save_freq='epoch', verbose=1,
                 MyModelCheckpoint(
                     filepath=model_dir + '/Checkpoint/Checkpoint--epoch={epoch:03d}--val_loss={val_loss:.4f}.h5',
                     save_freq='epoch', verbose=1,
                     save_weights_only=False)]
    cb_temp = ['Epoch Timing', 'Model Checkpoint']

    if not auto:
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
    else:
        for key in auto:
            callbacks.append(cb_dict.get(key))
            cb_temp.append(cb_names.get(key))

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
        base = sanitize_path(base) + '/'
        if not isdir(base):
            Path(base).mkdir(parents=True, exist_ok=True)
    path_dir = base + input(f'Input {target} directory: {base}')
    while (isdir(path_dir) and new) or (not isdir(path_dir) and not new):
        prompt = f'Directory {path_dir} exists!' if new else f'Directory {path_dir} does not exist!'
        print(prompt)
        path_dir = base + input(f'Input {target} directory: {base}')

    if new:
        Path(path_dir).mkdir()

    return sanitize_path(path_dir)


def opts_menu(txt: str, resp_dic: dict = None):
    resp_dic = {} if resp_dic is None else resp_dic
    options = findall(r'[(\[{](\w|\d+)[)\]}]', txt) or None
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


def dic_menu(dic: dict, prompt=''):
    if len(dic) == 1:
        return list(dic.values())[0]
    line_length = np.max([len(x) for x in dic.values()])
    line_length = np.max([line_length + 5, len(prompt), 50])
    prompt = prompt + f'\n' + '-' * line_length + '\n' + '\n'.join(
        [f'{i} {"-" * 5}> {j}' for i, j in dic.items()]) + '\n' + '-' * line_length + '\n'
    dic = dict(zip([str(x) for x in dic.keys()], dic.values()))

    result = dic.get(input(prompt), None)
    while result is None:
        print('Unexpected response.')
        result = dic.get(input(prompt), None)
    return result


def initiate_training(**kwargs):
    batch_size = kwargs.get('batch_size')
    batch_size = batch_size or get_nat('Batch size')
    epochs = kwargs.get('epochs')
    epochs = epochs or get_nat('Number of epochs')
    training_dir = sanitize_path(kwargs.get('training_dir'))
    training_dir = training_dir if training_dir and isdir(training_dir) else get_dir('training', new=False)
    dirs = {}
    keys = []
    keys1, keys2 = ['training', 'validation'], ['input', 'label']
    for key1 in keys1:
        for key2 in keys2:
            keys.append('_'.join([key1, key2]))
            for file in glob.glob('/'.join([training_dir, '*.npy'])):
                file = sanitize_path(file)
                if key2 in basename(file) and key1 in basename(file):
                    dirs.update({keys[-1]: file})
    if len(dirs) < 4:
        missing = [i for i in keys if i not in dirs.keys()]
        for key in missing:
            print(f'{key.replace("_", " ")} file not found.')
            temp_dir = get_file(key.replace('_', ' ') + ':')
            dirs.update({key: temp_dir})

    training_input = dirs.get('training_input')
    training_label = dirs.get('training_label')
    validation_input = dirs.get('validation_input')
    validation_label = dirs.get('validation_label')

    return [batch_size, epochs, (training_input, training_label), (validation_input, validation_label)]


def sanitize_path(path: str):
    if isinstance(path, str):
        path = path.replace('\\', '/')
        return path.removesuffix('/').removeprefix('/')
    return path


def create_cnn(metric=metrics.RootMeanSquaredError(), loss_func=losses.mse, optimizer=optimizers.Adadelta(),
               kernel_size=(3, 3), pool_size=(2, 2), **kwargs):
    model_dir = kwargs.get('model_name', '')
    model_dir = '/'.join(['models', model_dir]) if model_dir else model_dir
    if model_dir and not isdir(model_dir):
        model_dir = sanitize_path(model_dir)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    else:
        if model_dir:
            print(f'Directory {model_dir} exists!')
        model_dir = get_dir(target='model', new=True, base='models/')

    model_name = basename(model_dir)
    callbacks, callback_log = get_cbs(model_dir=model_dir, init_epoch=0, auto=kwargs.get('callback', False))
    batch_size, epochs, training, validation = initiate_training(**kwargs)
    num_sample, num_val, input_shape = validate_data(training=training, validation=validation)
    # input_training, label_training = training
    # input_validation, label_validation = validation

    ### Creating the model directory ###
    print('Creating model directory.')
    print('Preparing log file.')
    log = create_log(batch_size=batch_size, epochs=epochs, numsample=num_sample, numval=num_val,
                     input_training=training[0], input_validation=validation[0], label_training=training[1],
                     label_validation=validation[1], input_shape=input_shape, kernel_size=kernel_size,
                     pool_size=pool_size, loss_func=loss_func, optimizer=optimizer, metric=metric, first_run=True,
                     name=model_name)
    log.append(callback_log)
    comments = kwargs.get('comments')
    comments = comments or input('Additional Comments:')
    log.append(comments + '\n')
    write_log(model_dir=model_dir, log=log)
    print(f'Building model {model_name}.')

    model = create_model(loss_func=loss_func, optimizer=optimizer, metric=metric, path_dir=model_dir,
                         input_shape=input_shape, kernel_size=kernel_size, pool_size=pool_size)

    train_model(model=model, batch_size=batch_size, epochs=epochs, training=training, validation=validation,
                callbacks=callbacks, model_dir=model_dir)


def dir_menu(pattern: str, prompt: str, sanitize=''):
    dirs = [sanitize_path(x).removeprefix(sanitize) for x in glob.glob(pattern)]
    options = dict(zip(range(len(dirs)), dirs))

    if options:
        return dic_menu(dic=options, prompt=prompt)
    else:
        raise FileNotFoundError(f'No files/folders matching {pattern} found.')


def load_cnn(**kwargs):
    model_dir, model_name = (lambda x: (x.removesuffix('\\'), basename(x.removesuffix('\\'))))(
        dir_menu(pattern='models/*/', prompt='Choose model to load'))

    model_to_load = dir_menu(pattern=f'{model_dir}/*/*.h5', prompt='Choose checkpoint to load', sanitize=model_dir)

    batch_size, epochs, training, validation = initiate_training()
    num_sample, num_val, input_shape = validate_data(training=training, validation=validation)
    init_epoch = int(rmatch(r'.*epoch=(\d{3}).*', model_to_load).group(1))
    callbacks, callback_log = get_cbs(model_dir=model_dir, init_epoch=init_epoch)
    print('Preparing log file.')
    log = create_log(batch_size=batch_size, epochs=epochs, numsample=num_sample, numval=num_val,
                     input_training=training[0], input_validation=validation[0], label_training=training[1],
                     label_validation=validation[1], input_shape=input_shape, init_epoch=init_epoch,
                     model_to_load=model_to_load)
    log.append(callback_log)
    log.append(input('Additional Comments:') + '\n')
    write_log(model_dir=model_dir, log=log, insert=True)

    print(f'Loading {model_to_load} checkpoint of model {model_name}.')
    model = load_model(f'{model_dir}/{model_to_load}')

    train_model(model=model, batch_size=batch_size, epochs=epochs, training=training, validation=validation,
                callbacks=callbacks, model_dir=model_dir, init_epoch=init_epoch)


def write_log(model_dir, log, insert=False):
    if insert:
        temp = log
        log, insert_pos = fetch_log(model_dir)
        for line, i in zip(temp, range(insert_pos, insert_pos + len(temp))):
            log.insert(i, line)
    with open(f'{model_dir}/model.txt', 'wt') as file:
        print('Writing logs to file.')
        file.writelines(log)


def fetch_log(model_dir):
    with open(f'{model_dir}/model.txt', 'rt') as file:
        log = file.readlines()
    for i in range(len(log)):
        if 'Epoch\t\t\tTime' in log[i]:
            return [log, i - 1]
    raise ValueError('No insert position found in log file.\n' + ''.join(log))


def train_model(model: Sequential, batch_size, epochs, training, validation, callbacks, model_dir, init_epoch=0):
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
    Path(f'{model_dir}/History').mkdir(exist_ok=True)
    with open(f'{model_dir}/History/history_{date.replace("/", "-")}_{tm.replace(":", "")}.pickle', 'xb') as file:
        dump(history.history, file)
    write_log(model_dir=model_dir, log=f'{model.name} completed training sequence successfully.\n', insert=True)
    print(f'{model.name} has finished training successfully.')


def main():
    opts_menu('(L)oad/(C)reate model? ', {'c': create_cnn, 'l': load_cnn})()


if __name__ == '__main__':
    main()
