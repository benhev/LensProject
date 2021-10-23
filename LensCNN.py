import os
import glob
import inspect
import sys

import numpy as np
from tkinter import filedialog as filedlg, Tk
from tensorflow.keras.utils import Sequence as Keras_Sequence
from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D  # , Dropout
from tensorflow.keras import callbacks
from pathlib import Path
from os.path import isdir, isfile, basename, dirname
import time
from datetime import datetime
from pickle import dump
from re import compile as rcompile, match as rmatch, escape


class AutoCheckpoint(callbacks.ModelCheckpoint):
    """
    A ModelCheckpoint class which is pre-made using the directory scheme of this script.
    Saves checkpoint every epoch by default to model_dir/Checkpoint.
    This class also implements erasure of previous epoch checkpoint if one exists (see _save_model method).
    See also official TensorFlow/Keras documentation for more information on ModelCheckpoint and Callback classes.

    :param model_dir: Path to model folder, string.
    :param verbose: Print feedback to terminal, int: (0,1). 1 (on) by default.
    :param save_weights_only: Only save weights in checkpoint save, boolean. False by default.
    :param save_freq: How often to perform a save. 'epoch' by default. Integer will save every save_freq batches.
    :param options: See Tensorflow/Keras documentation
    :param kwargs: See Tensorflow/Keras documentation
    """

    def __init__(self, model_dir, verbose=1, save_weights_only=False, save_freq='epoch', options=None,
                 **kwargs):
        """
        Reimplementation of the constructor in order to override some parameters and set defaults which work with the directory scheme.
        """
        filepath = f'{model_dir}' + '/Checkpoint/Checkpoint--epoch={epoch}--val_loss={val_loss}.h5'
        super().__init__(filepath=filepath, verbose=verbose, options=options, save_freq=save_freq,
                         save_weights_only=save_weights_only, **kwargs)

    # This method has a signature issue. (See BestCheckpoint._save_model)
    def _save_model(self, epoch, logs=None, batch=None):
        """
        Reimplementation of _save_model of ModelCheckpoint in order to delete previous epoch save.
        Otherwise identical.

        :param epoch: Current epoch.
        :param logs: See TF/Keras documentation.
        :param batch: Current batch.
        """
        # regular expressions are used to find previous saves as the val_loss is unknown.
        regex = rcompile(
            escape(dirname(self.filepath)) + r'\\[cC]heckpoint[-_\w]*epoch=\d+[-_\s]*val_loss=\d+\.\d+\.h5')
        prev_epoch = [f for f in glob.glob(f'{dirname(self.filepath)}/*.h5') if
                      regex.match(f) and f'epoch={epoch}' in f]
        # epoch is in fact 1 less than the number of epoch running
        assert len(prev_epoch) <= 1, 'Found more than one previous epoch.'
        prev_epoch = prev_epoch[0] if len(prev_epoch) == 1 else None
        if prev_epoch:
            # Remove previous epoch save if one such exists.
            os.remove(prev_epoch)
        super()._save_model(epoch=epoch, logs=logs, batch=batch)


class BestCheckpoint(callbacks.ModelCheckpoint):
    """
    A ModelCheckpoint class which is pre-made using the directory scheme of this script.
    Saves only best seen checkpoint to model_dir/BestFit and logs the parameters to the model.txt file.
    See also official TensorFlow/Keras documentation for more information on ModelCheckpoint and Callback classes.

    :param model_dir: Path to model folder, string.
    :param monitor: Value to monitor, string. 'val_loss' by default.
    :param verbose: Print feedback to terminal, int: (0,1). 1 (on) by default.
    :param save_weights_only: Only save weights in checkpoint save, boolean. False by default.
    :param mode: Comparison method of different saves, string. 'min' by default.
    :param options: See TensorFlow/Keras documentation
    :param kwargs: See TensorFlow/Keras documentation
    """

    def __init__(self, model_dir, monitor='val_loss', verbose=1, save_weights_only=False,
                 mode='min', options=None, **kwargs):
        """
        Reimplementation of the constructor in order to override some parameters and set defaults which work with the directory scheme.
        """
        date, tm = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')).split()
        filepath = f'{model_dir}/BestFit/BestFit--{date.replace("/", "-")}_{tm.replace(":", "")}--' \
                   + 'epoch={epoch}--' + monitor + '={' + monitor + '}.h5'
        super().__init__(filepath=filepath, monitor=monitor, verbose=verbose, save_best_only=True,
                         save_weights_only=save_weights_only, mode=mode, options=options, **kwargs)

    # There is an issue with the signature of this method, for some reason it is dynamically called with an argument
    # which is undeclared in its signature (batch). Through trial and error I got this to work, it might require some
    # tinkering if tensorflow/keras is updated. This may also be an issue with conflicting definitions of the keras
    # ModelCheckpoint class, once in the keras standalone package and another in the tensorflow.keras module
    def _save_model(self, epoch, logs, batch=None):
        """
        Internally used method.
        An extension of existing method in ModelCheckpoint.
        Logs the save if such is performed to model.txt and call parent method.
        See TensorFlow/Keras documentation for additional information.

        :param epoch: Current epoch, int.
        :param logs: log, dict. See TF docs.
        """
        current = logs.get(self.monitor)
        log_dir = dirname(self.filepath).removesuffix('/BestFit') + '/model.txt'
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

            # Remove old saves whose monitored value is not better than the current. Ideally no more than 10 saves
            # exist at any time, but if they are all better than the current save it will not remove any. This is an
            # artifact of the naming scheme since each best checkpoint has an epoch and a monitor value so the save
            # will not overwrite (due to having different file names). This is also true for the date and time part
            # of the file name. A consequence of this is that a BestCheckpoint class will not delete unmonitored
            # saves nor count them as part of the 10 maximum to keep.
            files = glob.glob(f'{dirname(self.filepath)}/*{self.monitor}*.h5')
            if len(files) >= 10:
                regex = rcompile(escape(dirname(
                    self.filepath)) + r'\\[bB]est[fF]it[-_\w]*epoch=(\d+)[-_\s]*' + escape(
                    self.monitor) + r'=(\d+\.\d+)\.h5')
                num_to_remove = len(files) - 9
                rm_cands = []
                for f in files:
                    if regex.match(f):
                        rm_cands.append(sanitize_param(regex.match(f).groups()))
                rm_cands.sort(key=lambda x: x[1], reverse=True)
                rm_cands = rm_cands[:num_to_remove]
                for epch, monitor in rm_cands:
                    if self.monitor_op(current, monitor):
                        rm_file = [f for f in files if f'epoch={epch}' in f][0]
                        print('Removing', sanitize_path(rm_file).removeprefix(dirname(self.filepath) + '/'))
                        os.remove(rm_file)

        # After saving logs to file, call the parent method to finish proper ModelCheckpoint execution.
        # the function's signature has no batch argument but in runtime it receives it as a call
        # I've left this as it is because it works
        super()._save_model(epoch=epoch, logs=logs, batch=batch)


class TimeHistory(callbacks.Callback):
    """
    Epoch timing logger callback.
    This callback writes epoch times to model.txt

    :param model_dir: Path to model folder, string.
    :param initial_epoch: Starting epoch, int.
    """

    def __init__(self, model_dir, initial_epoch):
        self.dir = model_dir
        # self.name = name
        self.epoch = initial_epoch
        self.epoch_time_start = None
        super().__init__()

    def on_train_begin(self, logs=None):
        """
        Internally used method.
        Commands to be performed at beginning of training.

        :param logs: logs, dict. See TF documentation.
        """
        with open(f'{self.dir}/model.txt', 'at') as file:
            if not self.epoch:
                file.write('\n\n\tEpoch\t\t\tTime\t\t\tBest\n')
                file.write('=' * 100 + '\n')
            else:
                file.write('-' * 100 + '\n')

    def on_epoch_begin(self, epoch, logs=None):
        """
        Internally used method.
        Commands to be performed at beginning of each epoch.

        :param epoch: Current epoch, int.
        :param logs: logs, dict. See TF documentation.
        """
        # Start epoch timer
        self.epoch_time_start = time.time()
        self.epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        Internally used method.
        Commands to be performed at the end of each epoch.

        :param epoch: Current epoch, int.
        :param logs: logs, dict. See TF documentation.
        """
        # Log time elapsed to file
        with open(f'{self.dir}/model.txt', 'at') as file:
            file.write(f'\t{self.epoch}\t\t\t{time_convert(time.time() - self.epoch_time_start)}\n')


# Data is too large to store in memory all at once, this Sequence class handles the input data in batches.
class LensSequence(Keras_Sequence):
    """
    Keras data sequence which reads numpy files in chunks.

    :param x_set: Directory to input as *.npy file.
    :param y_set: Directory to labels as *.npy file.
    :param batch_size: Size of batches to be supplied to model.
    """

    def __init__(self, x_set, y_set, batch_size):
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

        perm = np.random.permutation(len(batch_x))

        return batch_x[perm], batch_y[perm]


def npy_get_shape(file: str):
    """
    Return the shape of a numpy array stored in *.npy file as iterable.

    :param file: File directory.
    :return: Shape of stored numpy array as numpy array.
    """
    with open(file, 'rb') as f:
        _, _ = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f)
    return shape


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
        # Start byte can get very large if the arrays are large
        # it's best to use int64
        start_byte = np.prod([start_row, row_size, dtype.itemsize], dtype='int64')
        file.seek(start_byte, 1)
        n_items = row_size * num_rows
        flat = np.fromfile(file, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])


# This function defines the CNN architecture, change it here.
def create_model(loss_func, optimizer, metric, model_dir: str, input_shape: tuple, kernel_size=(3, 3),
                 pool_size=(2, 2)):
    """
    Creates a model according to the architecture defined inside.

    :param loss_func: Loss function from tf.keras.losses (or custom or name according to TF docs).
    :param optimizer: Optimizer to use from tf.keras.optimizers (or custom or name according to TF docs).
    :param metric: Metrics to use (as a list) from tf.keras.metrics (or custom or name(s) according to TF docs).
    :param model_dir: Path to model folder, string.
    :param kernel_size: Convolution kernel size, double. (3,3) unless otherwise specified.
    :param pool_size: UpSampling and MaxPooling pool size, double. (2,2) unless otherwise specified.
    :param input_shape: Input image shape, triple: (W,H,D).
    :return: Sequetial Keras model built according to the specified architecture.
    """
    # If the model is over-fitting it may be beneficial to introduce some Dropout layer
    # Further one can play with the activation functions to get different results
    model = Sequential(name=basename(model_dir))

    model.add(Conv2D(16, kernel_size=kernel_size, activation='relu', input_shape=input_shape, padding='same',
                     kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=pool_size, padding='same'))

    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=pool_size, padding='same'))

    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'))

    model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metric)

    with open(f'{model_dir}/summary.txt', 'wt') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    return model


def time_convert(seconds):
    """
    Convert time from seconds to regular hh:mm:ss format.

    :param seconds: Time in seconds, float.
    """
    # This function is necessary for time logging to function
    hrs = np.floor(seconds / 3600).astype('int')
    seconds = np.mod(seconds, 3600)
    mins = np.floor(seconds / 60).astype('int')
    seconds = np.mod(seconds, 60).astype('int')

    return f'{hrs:02d}:{mins:02d}:{seconds:02d}'


def create_log(batch_size: int, epochs: int, numsample: int, numval: int, input_training: str, input_validation: str,
               label_training: str, label_validation: str, input_shape, kernel_size=(3, 3),
               pool_size=(2, 2), loss_func=None, optimizer=None, metric=None, first_run=False, name: str = None,
               init_epoch: int = None,
               model_to_load=None):
    """
    Creates a log entry, whether for an existing file or for a first run

    :param batch_size: Batch size, integer.
    :param epochs: Number of epochs, integer.
    :param numsample: Number of training samples, integer.
    :param numval: Number of validation samples, integer.
    :param input_training: Training input path, string.
    :param input_validation: Validation input path, string.
    :param label_training: Training label path, string.
    :param label_validation: Validation label path, string.
    :param first_run: Creates a new log for new models along with the log header, boolean. False by default.

    Required only if first_run is True:
        :param input_shape: Shape of input images, (W,H,D) tuple.
        :param kernel_size: Shape of convolution kernels, double. (3,3) by default.
        :param pool_size: Size of MaxPooling and UpSampling kernels, double. (2,2) by default
        :param loss_func: Custom loss function or tf.keras.losses function or identifying string (See TF docs).
        :param optimizer: Custom optimizer function or tf.keras.optimizers function or identifying string (See TF docs).
        :param metric: Custom metric functions (as list) or tf.keras.metrics function or identifying string (See TF docs).
        :param name: Model name, string.

    Required only if first_run is False:
        :param init_epoch: Initial epoch of training instance, int.
        :param model_to_load: Directory of model in case of existing model, str. Used to log the name of the loaded checkpoint.

    :return: Log string to be written to file.

    """
    date, tm = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')).split()
    if first_run and all([loss_func, name, optimizer, metric]):
        temp = [f'Model {name}\n',
                f'Loss Function: {loss_func if isinstance(loss_func, str) else loss_func.__name__}\n',
                f'Optimizer: {optimizer if isinstance(optimizer, str) else optimizer._name}\n',
                f'Metrics: ' + ', '.join([x if isinstance(x, str) else x.name for x in metric]) + '\n',
                f'Conv. kernel size: {kernel_size}\nMax and UpSampling pool size: {pool_size}\n',
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


def sanitize_param(param):
    """
    Sanitizes input parameters for dynamic function calling.

    :param param: Some parameter, string. Usually obtained from user input.
                  If an iterable is supplied will sanitize each entry and return as list.
    :return: param as a natural type.
    """
    if isinstance(param, str):
        try:
            param = int(param)
        except ValueError:
            try:
                param = float(param)
            except ValueError:
                pass
        return param
    elif len(param) > 1:
        temp = []
        for x in param:
            temp.append(sanitize_param(x))
        return temp
    else:
        raise TypeError(f'{param} is neither a string nor iterable.')


# TODO docs
def get_cbs(model_dir, epochs, batch_size, init_epoch, auto=None):
    """
    Function to generate and return callbacks and a log message containing callback names.

    :param model_dir: Path to model folder, string.
    :param epochs: Number of epochs, int.
    :param batch_size: Batch size, int.
    :param init_epoch: Initial epoch of training instance, int.
    :param auto: Can be supplied as a list or dictionary.
    :return:
    """

    def get_args(func):
        """
        This function retrieves arguments of a callable from its signature and returns a dictionary of arguments.
        Arguments which do not have default values are returned with a value of '' (empty string).

        :param func: Some callable object.
        :return: Dictionary of arguments, dict.
        """
        sig = inspect.signature(func)
        args = {}
        for param in sig.parameters.values():
            args.update({param.name: '' if param.default is param.empty else param.default})
        # Not interested in args and kwargs (good enough for callbacks)
        args.pop('args', None)
        args.pop('kwargs', None)
        return args

    def check_parents(test_class, test_parents):
        if not isinstance(test_parents, list):
            test_parents = [test_parents]
        for parent in test_class.__bases__:
            if parent in test_parents or check_parents(parent, test_parents):
                return True
        return False

    def set_param(dic: dict, k=None, required=False, new=False):
        nonlocal model_dir, model_name, epochs, batch_size, init_epoch
        if not new and k not in dic.keys():
            raise KeyError(f'Key {k} not found.')
        if new:
            required = False
            k = input('Input keyword to set:')
            if k == '':
                return
            gss = [gkey for gkey in sorted(guess_dic.keys()) if gkey in k]
            if gss:
                gss = k if k in gss else gss[0]
                print(f'Auto suggesting value for parameter {k}:')
                dic.update({k: guess_dic.get(gss)})
        prompt = f'Changing value of {k}. Input as PYTHON EXPRESSION. \n' \
                 + ('This is a mandatory field.\n' if required else 'Leave blank to leave field unchanged.\n') \
                 + f'{k}={dic.get(k, "")}\n'

        success = False
        while not success:
            print(prompt)
            print('Useful parameters in the scope:')
            print(', '.join(get_args(get_cbs).keys()) + ', model_name\n\n')

            user_inp = input(f'{k}=')
            try:
                user_inp = eval(user_inp)
            except:
                if user_inp != '':
                    print(f'Could not interpret expression: {user_inp}.')
                elif user_inp == '' and not required:
                    success = True
            else:
                dic.update({k: user_inp})
                success = True
            finally:
                pass

    def get_verify_cb():
        add = False
        temp_cb = dic_menu(dict(zip(range(1, 1 + len(options)), options.keys())), prompt=cb_prompt,
                           quit_option={'q': ''})
        while temp_cb and not add:
            # temp_cb = options.get(temp_cb)
            print(
                '=' * 100 + f'\nShowing documentation for {options.get(temp_cb).__name__} callback.\nEnsure all required arguments are met.\n' + '=' * 100)
            # time.sleep(2)
            print(options.get(temp_cb).__doc__)
            print('=' * 100)
            add = (input('Add callback? (Y/N)\n').lower() == 'y')
            if not add:
                temp_cb = dic_menu(dict(zip(range(1, 1 + len(options)), options.keys())), prompt=cb_prompt,
                                   quit_option={'q': ''})

        return temp_cb if temp_cb == '' else options.pop(temp_cb, None)

    model_name = basename(model_dir)
    keras_options = {k: v for k, v in inspect.getmembers(callbacks, inspect.isclass) if
                     callbacks.Callback in v.__bases__}
    options = {k: v for k, v in inspect.getmembers(sys.modules[__name__], inspect.isclass) if
               check_parents(v, [callbacks.Callback]) and 'keras.callbacks' not in v.__module__}
    options.update(keras_options)
    for key in ['AutoCheckpoint', 'ModelCheckpoint', 'TimeHistory']:
        options.pop(key, None)

    callback_list = [TimeHistory(model_dir=model_dir, initial_epoch=init_epoch), AutoCheckpoint(model_dir=model_dir)]
    callback_names = ['TimeHistory', 'AutoCheckpoint']
    if auto is None:
        guess_dic = {
            'log_dir': f'logs/{model_name}',
            'path': model_dir,
            'dir': model_dir,
            'name': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'init': init_epoch,
            'initial_epoch': init_epoch,
        }
        cb_prompt = f'\n\nSelect callbacks to utilize.\nNote: time logging and model checkpoint are on by default and ' \
                    f'cannot be turned off\nas they are necessary for epoch timing and model saving (' \
                    f'respectively).\nEnter q to quit this menu.'
        cb = get_verify_cb()
        while cb:
            # options_menu.pop(cb)
            # cb = options.get(cb)
            args = get_args(cb)
            req_keys = [k for k, v in args.items() if v == '']
            for key in req_keys:
                guess = [gkey for gkey in sorted(guess_dic.keys()) if gkey in key]
                if guess:
                    guess = key if key in guess else guess[0]
                    print(f'Auto suggesting value for required parameter {key}:')
                    args.update({key: guess_dic.get(guess)})
                    set_param(args, key)
                else:
                    set_param(args, k=key, required=True)

            key_prompt = f'Current parameter values for {cb.__name__} are listed.\nChoose parameter to ' \
                         + 'modify or press Enter to approve and continue.\n' \
                         + 'Type kwargs to input new keyword argument or DELETE to delete.'
            quit_options = {'': 'approve', 'kwargs': 'kwargs', 'DELETE': 'delete'}
            temp_key = dic_menu(args, prompt=key_prompt, key=True, quit_option=quit_options)
            while temp_key:
                if temp_key == 'kwargs':
                    set_param(args, new=True)
                elif temp_key == 'DELETE':
                    del_key = input('Input parameter to delete:')
                    if del_key in req_keys:
                        print('Required parameters cannot be deleted.')
                    else:
                        args.pop(del_key, None)
                else:
                    set_param(args, k=temp_key)

                temp_key = dic_menu(args, prompt=key_prompt, key=True, quit_option=quit_options)

            callback_list.append(cb(**args))
            callback_names.append(cb.__name__)
            print(f'Added callback {cb.__name__}.\n{len(callback_list)} callbacks enabled.')
            cb = get_verify_cb()

    else:
        # The function may have been called externally to this script and can be supplied with arguments
        # in order to circumvent the need for user input
        for name in auto:
            cb = options.get(name)
            callback_list.append(cb(**auto.get(name)))
            callback_names.append(cb.__name__)
            print(f'Added callback {cb.__name__}.\n{len(callback_list)} callbacks enabled.')

    return callback_list, 'Callbacks: ' + ', '.join(callback_names) + '\n\n'


def get_dir_gui(base: str = './', title='Select folder', file=False):
    """
    Get an existing directory via gui interface.

    :param base: Initial path of prompt, string.
    :param title: Window title, string.
    :param file: Get file instead of directory, boolean. False by default.
    :return: Sanitized path as string.
    """
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    filepath = ''
    while not filepath:
        if file:
            filepath = filedlg.askopenfilename(initialdir=base, title=title)
        else:
            filepath = filedlg.askdirectory(initialdir=base, title=title)
    root.destroy()
    return sanitize_path(filepath).removeprefix(sanitize_path(os.getcwd()) + '/')


def get_nat(name: str, limit: int = np.inf):
    """
    Gets a natural number from the user.

    :param name: Name of parameter to request, str.
    :param limit: (Optional) Largest acceptable number, int.
    :return: Natural number from user as integer.
    """

    def isnat(test):
        """
        Returns True if test is an integer larger than 0 (Natural), and False otherwise.

        :param test: object to test, any.
        :return: Boolean.
        """
        try:
            return True if int(test) > 0 else False
        except ValueError:
            return False

    nat = input('Enter a positive integer' + (f' no larger than {limit}' if limit != np.inf else '') + f' for {name}: ')
    while not isnat(nat) or (isnat(nat) and int(nat) > limit):
        nat = input(f'{name} should be a positive integer' + (
            f' no larger than {limit}.' if limit != np.inf else '.') + f'\n{name}: ')
    return int(nat)


def validate_data(training: tuple, validation: tuple):
    """
    Validates that the input and label data are of matching shapes

    :param training: Tuple of directories to training data in the form (input, label)
    :param validation: Tuple of directories to validation data in the form (input, label)
    :return: List of: number of training instances(int),number of validation instances(int), input shape (H,W,D) (tuple)
    """
    tr_inp_shape, tr_lab_shape = list(map(npy_get_shape, training))
    val_inp_shape, val_lab_shape = list(map(npy_get_shape, validation))

    assert tr_inp_shape == tr_lab_shape, 'Shapes of training input and labels must match!'
    assert val_inp_shape == val_lab_shape, 'Shapes of validation input and labels must match!'

    return [tr_inp_shape[0], val_inp_shape[0], tr_inp_shape[1:]]


def get_dir(target, new: bool, base=''):
    """
    Requests a directory from the user.

    :param target: The target of the directory, some name describing the request, string.
    :param new: Should the directory exist (False) or not (True), boolean.
    :param base: Underlying path in which to create the new directory, string.
                 If path of base does not one will be created.
    :return: Sanitized directory as string.
    """
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


def dic_menu(dic: dict, prompt='', pop=False, key=False, quit_option: dict = None):
    """
    Creates a UI menu from dictionary.

    :param dic: Dictionary containing menu items in the form {label:action}, dict. Keys should be castable to string.
    :param prompt: Prompt to show the user, string. Empty by default.
    :param pop: Pop the chosen option from the dictionary, boolean. False by default.
    :param key: Return key instead of value, boolean. False by default.
    :param quit_option: (Optional) Actions which will not be shown to the user but will be accepted regardless.
                        This argument is passed as a dictionary in the same form as other menu items:

                                quit_option={label:action}

                        It will not be shown to the user, take care to incorporate its functionality into your prompt.
                        Return logic follows that of other options (with regard to the 'key' argument).
                        This argument may contain several options.

                                quit_option={label1:action1,label2:action2,...}
    :return: Key if 'key'=True, otherwise the matching value from the dictionary.
    """
    if quit_option is None and len(dic) == 1:
        # return the only option as long as there isn't a quit option
        return list(dic.keys() if key else dic.values())[0]
    # calculate the length of the menu for aesthetic purposes
    line_length = np.max([len(f'{x}') for x in dic.values()] or 50)
    promp_length = np.max([len(x) for x in prompt.split('\n')])
    line_length = np.max([line_length + 10, promp_length, 50])
    # prepare prompt to user
    prompt = prompt + f'\n' + '-' * line_length + '\n' + '\n'.join(
        [f'{i} {"-" * 5}> {j}' for i, j in dic.items()]) + '\n' + '-' * line_length + '\n'
    # change keys to str so that user input can be tested against them
    temp_dic = {str(k): v for k, v in dic.items()}
    if quit_option:
        # add quit options. note these will not be displayed as the prompt has already been established
        temp_dic.update({str(k): (v if v is not None else '') for k, v in quit_option.items()})
    # Finally get the user input and check that such a key exists in the dictionary
    user_inp = input(prompt)
    result = temp_dic.get(user_inp, None)
    while result is None:
        print('Unexpected response.')
        user_inp = input(prompt)
        result = temp_dic.get(user_inp, None)
    if pop:
        dic.pop(sanitize_param(user_inp), None)

    return sanitize_param(user_inp) if key else result


def get_training_files(training_dir: str, validation=True):
    """
    The function searches for files labelled with training/validation and input/label.
    It will attempt to find the proper keywords in the file name and assign that file to the proper dictionary key in the format: (training/validation)_(input/label).
    Files which were not found will be requested from the user.

    :param training_dir: Training directory containing *.npy files, string.
    :param validation: Search for validation data, bool. True by default.
    :return: dictionary of paths by combinations of keys.
    """
    dirs = {}
    keys = []
    keys1 = ['training', 'validation'] if validation else ['training']
    keys2 = ['input', 'label']

    for key1 in keys1:
        for key2 in keys2:
            # Make a list with all possible combinations
            keys.append('_'.join([key1, key2]))
            for file in glob.glob('/'.join([training_dir, '*.npy'])):
                # Check each key against the directory and add to the dictionary if a match is found
                file = sanitize_path(file)
                if key2 in basename(file) and key1 in basename(file):
                    dirs.update({keys[-1]: file})
    if len(dirs) < 4:
        # Find which files were not found from the automatic file fetching
        missing = [i for i in keys if i not in dirs.keys()]
        for key in missing:
            # request the file from the user
            print(f'{key.replace("_", " ")} file not found.')
            temp_dir = get_dir_gui(title=key.replace('_', ' '), file=True)
            dirs.update({key: temp_dir})

    return dirs


def initiate_training(**kwargs):
    """
    Gets initial meta-parameters of training session.
    May be called with keyword arguments or nothing at all.
    Whichever kwarg is supplied and valid will not be requested from the user.
    
    :batch_size: Batch size, int.
    :epochs: Number of epochs to run, int.
    :training_dir: Training directory containing numpy files with training and validation data, string.
    :return: List of batch size, number of epochs, (training input path, training label path), (validation input path, validation label path)
    """
    batch_size = kwargs.get('batch_size')
    batch_size = batch_size or get_nat('Batch size')
    epochs = kwargs.get('epochs')
    epochs = epochs or get_nat('Number of epochs')
    training_dir = sanitize_path(kwargs.get('training_dir'))
    # training_dir = training_dir if training_dir and isdir(training_dir) else get_dir('training', new=False)
    print('Select training directory')
    training_dir = training_dir if training_dir and isdir(training_dir) else get_dir_gui(
        title='Select training directory')

    dirs = get_training_files(training_dir=training_dir)

    training_input = dirs.get('training_input')
    training_label = dirs.get('training_label')
    validation_input = dirs.get('validation_input')
    validation_label = dirs.get('validation_label')

    return [batch_size, epochs, (training_input, training_label), (validation_input, validation_label)]


def sanitize_path(path: str):
    """
    Sanitizes a path such that any back slashes are transformed to front slashes.
    Slashes are also removed from the ends of the path should they exist.

    :param path: Path to sanitize, string.
    :return: Sanitized path as string of the form:
                \\folder1\\folder2/folder3/ --> folder1/folder2/folder3
    """
    if isinstance(path, str):
        path = path.replace('\\', '/')
        return path.strip(' /')
    return path


def dir_menu(pattern: str, prompt: str, sanitize=''):
    """
    User input menu for directories. Lists directories/files corresponding to pattern.
    Raises FileNotFoundError if no files/folders matching pattern are found.

    :param pattern: glob.glob compatible pattern, string.
    :param prompt: Prompt to show user, string.
    :param sanitize: (Optional) Directories to remove from the left, string.
    :return: Directory of existing path as string.
    """
    sanitize = sanitize_path(sanitize) + '/' if sanitize else sanitize
    dirs = [sanitize_path(x).removeprefix(sanitize) for x in glob.glob(pattern)]
    options = dict(zip(range(len(dirs)), dirs))

    if options:
        return dic_menu(dic=options, prompt=prompt)
    else:
        raise FileNotFoundError(f'No files/folders matching {pattern} found.')


def create_cnn(metric=None, loss_func=losses.mse, optimizer=optimizers.Adadelta(),
               kernel_size=(3, 3), pool_size=(2, 2), **kwargs):
    """
    Creates a new convolutional neural network.
    If insufficient kwargs supplied will default to user input.

    :param metric: Metrics to use, list. May be a list of names according to TF docs or proper metric functions
                   (custom or tf.keras.metrics). RMSE by default.
    :param loss_func: Loss function to use. May be a name string according to TF docs or a proper loss function
                      (custom or tf.keras.losses). MSE by default.
    :param optimizer: Optimizer to user. May be a name string according to TF docs or a proper optimizer object
                      (tf.keras.optimizers). AdaDelta by default.
    :param kernel_size: Convolution kernel size, double. (3,3) by default.
    :param pool_size: MaxPool and UpSampling kernel size, double. (2,2) by default.

    Additional optional key-word arguments:
        General:
            :model_dir: Path to model folder, string. Directory must not already exist.
            :callback: Callbacks in the format accepted in the 'auto' parameter in get_cbs(), list or dict.
            :comments: Comments to add to the log file, string.

        Relevant to initialize_training() (see function documentation):
            :batch_size:
            :epochs:
            :training_dir:
    """
    metric = metric or [metrics.RootMeanSquaredError()]
    model_dir = kwargs.get('model_dir', '')
    # model_dir = '/'.join(['models', model_dir]) if model_dir else model_dir
    if model_dir and not isdir(model_dir):
        model_dir = sanitize_path(model_dir)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    else:
        if model_dir:
            print(f'Directory {model_dir} exists!')
        print('Select model base directory.')
        model_dir = get_dir_gui(title='Select model base directory')
        model_dir = get_dir(target='model', new=True, base=model_dir)

    model_name = basename(model_dir)
    batch_size, epochs, training, validation = initiate_training(**kwargs)
    callbacks, callback_log = get_cbs(model_dir=model_dir, epochs=epochs, batch_size=batch_size, init_epoch=0,
                                      auto=kwargs.get('callback', None))
    num_sample, num_val, input_shape = validate_data(training=training, validation=validation)

    ### Creating the model directory ###
    print('Creating model directory.')
    print('Preparing log file.')
    log = create_log(batch_size=batch_size, epochs=epochs, numsample=num_sample, numval=num_val,
                     input_training=training[0], input_validation=validation[0], label_training=training[1],
                     label_validation=validation[1], input_shape=input_shape, kernel_size=kernel_size,
                     pool_size=pool_size, loss_func=loss_func, optimizer=optimizer, metric=metric, first_run=True,
                     name=model_name)
    log.append(callback_log)
    comments = kwargs.get('comments', None)
    comments = comments if comments is not None else input('Additional Comments:')
    log.append(comments + '\n')
    write_log(model_dir=model_dir, log=log)
    print(f'Building model {model_name}.')

    model = create_model(loss_func=loss_func, optimizer=optimizer, metric=metric, model_dir=model_dir,
                         input_shape=input_shape, kernel_size=kernel_size, pool_size=pool_size)

    train_model(model=model, batch_size=batch_size, epochs=epochs, training=training, validation=validation,
                callbacks=callbacks, model_dir=model_dir)


def load_cnn(**kwargs):
    """
    Loads an existing convolutional neural network save.
    If insufficient kwargs supplied will default to user input.

    Optional key-word arguments:
        General:
            :model_dir: Path to model folder, string. Directory must exist.
            :save_type: Type of save to load, string. Possible values: Checkpoint/cp or BestFit/bf/bst (case insensitive).
                        Must be supplied along model_name.
            :save_epoch: Epoch of save, int. Required for loading a best fit model (and sometimes a checkpoint if more than one exists).
                         save_epoch identifies the save and must be supplied.
            :callback: Callbacks in the format accepted in the 'auto' parameter in get_cbs(), list or dict.
            :comments: Comments to add to the log file, string.

        Relevant to initialize_training() (see function documentation):
            :batch_size:
            :epochs:
            :training_dir:
    """
    save_type = kwargs.get('save_type', '')
    model_dir = kwargs.get('model_dir', '')
    load_epoch = kwargs.get('save_epoch', '')
    model_to_load = ''

    if model_dir and save_type:
        if not isdir(model_dir):
            raise FileNotFoundError(f'Directory {model_dir} does not exist')
        model_name = basename(model_dir)
        if save_type.lower() in ['checkpoint', 'cp']:
            model_to_load = glob.glob(f'{model_dir}/Checkpoint/*.h5')
            if len(model_to_load) > 1:
                assert load_epoch, f'More than one checkpoint found in {model_dir}. Specify save_epoch to specify model to load.'
                model_to_load = [x for x in model_to_load if f'epoch={load_epoch:03d}' in x]
                if len(model_to_load) != 1:
                    print(
                        f'There should be no more than one checkpoint with a specific epoch in {model_dir}, found {len(model_to_load)}.')
                    model_to_load = []
        elif save_type.lower() in ['bestfit', 'bf', 'bst']:
            assert load_epoch, 'Cannot auto-load best fit without specified epoch.'
            model_to_load = glob.glob(f'{model_dir}/BestFit/*epoch={load_epoch:03d}*.h5')
            if len(model_to_load) != 1:
                print(
                    f'Cannot find specific best fit model for {model_name} with epoch={load_epoch:03d} in {model_dir}.')
                model_to_load = []
        else:
            raise ValueError(f'Unrecognized save_type: {save_type}')
        model_to_load = sanitize_path(model_to_load[0]).removeprefix(model_dir + '/') if model_to_load else None
    if not model_to_load:
        if model_to_load is None:
            print('Model not found!')
        print('Select model folder.')
        model_dir = get_dir_gui(title='Select model folder')
        model_name = basename(model_dir)
        # Get model to load from possible saves in model_dir
        model_to_load = dir_menu(pattern=f'{model_dir}/*/*.h5', prompt='Choose checkpoint to load', sanitize=model_dir)

    init_epoch = int(rmatch(r'.*epoch=(\d+).*', model_to_load).group(1))
    batch_size, epochs, training, validation = initiate_training(**kwargs)
    callbacks, callback_log = get_cbs(model_dir=model_dir, epochs=epochs, batch_size=batch_size, init_epoch=init_epoch,
                                      auto=kwargs.get('callback', None))
    num_sample, num_val, input_shape = validate_data(training=training, validation=validation)
    # get init_epoch from save name
    print('Preparing log file.')
    log = create_log(batch_size=batch_size, epochs=epochs, numsample=num_sample, numval=num_val,
                     input_training=training[0], input_validation=validation[0], label_training=training[1],
                     label_validation=validation[1], input_shape=input_shape, init_epoch=init_epoch,
                     model_to_load=model_to_load)
    log.append(callback_log)
    comments = kwargs.get('comments', None)
    comments = comments if comments is not None else input('Additional Comments:')
    log.append(comments)
    write_log(model_dir=model_dir, log=log, insert=True)

    print(f'Loading {basename(model_to_load)} save of model {model_name}.')
    model = load_model(f'{model_dir}/{model_to_load}')

    train_model(model=model, batch_size=batch_size, epochs=epochs, training=training, validation=validation,
                callbacks=callbacks, model_dir=model_dir, init_epoch=init_epoch)


def write_log(model_dir, log, insert=False):
    """
    Writes logs to model.txt in given directory.

    :param model_dir: Directory in which to log, string.
    :param log: Text to write to log, string/list of strings.
    :param insert: Insert to end of log (right before epoch timing), boolean. False by default.
    """
    if insert:
        temp = log
        log, insert_pos = fetch_log(model_dir)
        for line, i in zip(temp, range(insert_pos, insert_pos + len(temp))):
            log.insert(i, line)
    with open(f'{model_dir}/model.txt', 'wt') as file:
        print('Writing logs to file.')
        file.writelines(log)


def fetch_log(model_dir):
    """
    Fetches logs and insertion index from existing model.txt log file.
    Raises ValueError if insert_position is not found.

    :param model_dir: Directory containing model.txt
    :return: List: logs text(string), insert position(int)
             Insert position is one line above the epoch timing table.
    """
    with open(f'{model_dir}/model.txt', 'rt') as file:
        log = file.readlines()
    for i in range(len(log)):
        if 'Epoch\t\t\tTime' in log[i]:
            return [log, i - 1]
    raise ValueError('No insert position found in log file.\n' + ''.join(log))


def train_model(model: Sequential, batch_size, epochs, training, validation, callbacks, model_dir, init_epoch=0):
    """
    Trains the model and pickles the history.

    :param model: Model to be trained, keras Sequential model.
    :param batch_size: Batch size, int.
    :param epochs: Number of epochs, int.
    :param training: Training files, double: (input,label).
    :param validation: Validation files, double: (input,label).
    :param callbacks: List of callbacks, list.
    :param model_dir: Model directory, string.
    :param init_epoch: Initial epoch, int. 0 by default.
    """
    now = str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    date, tm = now.split()

    input_training, label_training = training
    input_validation, label_validation = validation
    # initialize a training sequence
    train_sequence = LensSequence(x_set=input_training, y_set=label_training, batch_size=batch_size)
    # initialize a validation sequence
    test_sequence = LensSequence(x_set=input_validation, y_set=label_validation, batch_size=batch_size)
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
    # Possible actions are load/create model or quit.
    actions = {'c': create_cnn, 'l': load_cnn, '': print}
    actions.get(
        dic_menu(prompt='Load/Create model?\nEnter to exit.', dic={'c': 'Create', 'l': 'Load'}, quit_option={'': ''},
                 key=True))()


if __name__ == '__main__':
    main()
