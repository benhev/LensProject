from lenstronomy.Util.param_util import phi_q2_ellipticity as phiq2el
from lenstronomy.Util.param_util import ellipticity2phi_q as el2phiq
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lenstronomy.Util.util import make_grid, array2image, make_grid_with_coordtransform
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
import numpy as np
from os.path import isfile, isdir, dirname
from pathlib import Path

from LensCNN import get_dir_gui

# import winsound

# FREQ = 1000  # Set Frequency To 2500 Hertz
# DUR = 500  # Set Duration To 1000 ms == 1 second

MARGIN_FRACTION = 0.4
VAL_SPLIT = 0.1


# def dist_check(lens_center, prev_centers, rng):
#     prev_centers = np.array(prev_centers)
#
#     # finish distance checking so two lenses are not too close together
#     for center in prev_centers:
#         if np.linalg.norm(center - lens_center) < rng:
#             return True
#     return False  # True if len(prev_centers) == 0 else False


def npy_write(filename: str, start_row, arr, size=None):
    """
    Custom function to write to an existing numpy file.
    Based on npy_write from LensCNN (See docs therein for author credits).

    :param filename: File to which the function will write, string.
    :param start_row: Initial position in which to start filling data, int.
                      This corresponds to the array index if the entire file was loaded into memory.
    :param arr: Array to write, np.ndarray.
    :param size: (Optional) Shape of numpy array to create, tuple. Only required if the file does not exist.
                            If file does not exist, will attempt to create one, in which case size must be given.
    :return: Index immediately after end of data, int.
    """
    print(f'Writing to {filename} at {start_row}')
    # This function is a hack and does not use documented behavior, though it works.
    assert start_row >= 0 and isinstance(arr, np.ndarray)
    num_rows = len(arr) if len(arr.shape) > 1 else 1
    if not isfile(filename):
        assert filename.endswith('.npy'), 'File name must have extension .npy.'
        if size is not None and num_rows <= size:
            if not isdir(dirname(filename)):
                Path(dirname(filename)).mkdir(parents=True, exist_ok=True)
            np.save(filename, np.zeros((size,) + arr.shape[1:], dtype=arr.dtype), allow_pickle=False)
        else:
            raise ValueError(
                'Size must be given as an argument when first creating file.\nSize must be >= length of arr to write.')
    with open(filename, 'rb+') as file:
        _, _ = np.lib.format.read_magic(file)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(file)
        assert not fortran, "Fortran order arrays not supported"
        # Make sure the offsets aren't invalid.
        if arr.dtype is not dtype:
            arr = arr.astype(dtype=dtype)
            print('Array data type is inconsistent with file. Casting to file data type.')
        assert start_row < shape[0], 'start_row is beyond end of file'
        if len(arr.shape) > 1:
            assert arr.shape[1:] == shape[1:]
        else:
            assert arr.shape == shape[1:]
        assert start_row + num_rows <= shape[0]
        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        row_size = np.prod(shape[1:])
        start_byte = np.prod([start_row, row_size, dtype.itemsize], dtype='int64')
        file.seek(start_byte, 1)
        arr.tofile(file)
    return start_row + num_rows


def grid(npix, deltapix, origin_ra, origin_dec):
    """
    Creates a PixelGrid class.

    :param npix: Number of pixels in each dimension, int.
    :param deltapix: Pixel resolution (arcsec/pixel), float.
    :param origin_ra: Right Ascension of origin (in arcsec), float.
    :param origin_dec: Declination of origin (in arcsec), float.
    :return: instance of PixelGrid class.
    """
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, p2a_transform, _ = make_grid_with_coordtransform(numPix=npix,
                                                                                          deltapix=deltapix,
                                                                                          center_ra=origin_ra,
                                                                                          center_dec=origin_dec,
                                                                                          subgrid_res=1,
                                                                                          inverse=False)
    kwargs = {'nx': npix, 'ny': npix, 'ra_at_xy_0': ra_at_xy_0,
              'dec_at_xy_0': dec_at_xy_0, 'transform_pix2angle': p2a_transform}
    return PixelGrid(**kwargs)


def generate_lens(grid_class):
    """
    Generates a lens instance with random position and ellipticity.

    :param grid_class: A PixelGrid class on which to place the lens.
    :return: A LensModel instance and a list of keyword arguments for each lens grouped in a tuple.
    """
    num_of_lenses = np.random.randint(1, 5)
    kwargs = []
    centers = []
    for _ in range(num_of_lenses):
        e1, e2 = phiq2el(np.random.uniform(0, np.pi), np.random.uniform(0.7, 1.0))
        theta = np.random.normal(loc=1, scale=0.01)
        center = sample_center(grid_class)
        centers.append(center)
        temp_kwargs = {'center_x': center[0], 'center_y': center[1], 'theta_E': theta, 'e1': e1, 'e2': e2}
        kwargs.append(temp_kwargs)

    return [LensModel(lens_model_list=['SIE'] * num_of_lenses), kwargs]


def contains(txt: str, substr: list):
    """
    Checks if any of a list of keywords is found in a given string.

    :param txt: Text to check against, string.
    :param substr: String(s) to check for, list of strings.
    :return: True if any substring is found.
    """
    for line in substr:
        if not isinstance(line, str):
            raise ValueError(f'{substr} is not a list of strings.')
        if line.lower() in txt.lower():
            return True
    return False


def sample_center(grid_class, margin=MARGIN_FRACTION):
    """
    Generates a random point in the grid, taking margins into account.
    Used to randomize center points for generated lenses.

    :param grid_class: An instance of the PixelGrid class.
    :param margin: (Optional) Forbidden image margins as fraction of the total dimension, float.
    :return: 2D vector object, as np.ndarray.
    """
    assert margin <= 1 / 2, 'MARGIN_FRACTION must be <=1/2.'
    xmin, ymin = (1 - 2 * margin) * np.array(list(map(np.min, grid_class.pixel_coordinates)))
    xmax, ymax = (1 - 2 * margin) * np.array(list(map(np.max, grid_class.pixel_coordinates)))
    # grid_len = len(grid_class.pixel_coordinates[0])
    # margin = np.floor(MARGIN_FRACTION * grid_len)
    cx, cy = list(map(lambda x: np.random.uniform(*x), [(xmin, xmax), (ymin, ymax)]))
    # cx, cy = np.around(grid_class.map_pix2coord(x=cx, y=cy), decimals=1)
    return np.array([cx, cy])


def make_image(data, names, kwargs_lens, extent, save_dir, lens_num=None):
    """
    Generates an image of the lensing scenario.
    Plots all data supplied, convergence is plotted with log scaling.

    :param data: Image data to plot, list of images.
    :param names: Names of plots, in order and of the same dimension as data, list of strings.
    :param kwargs_lens: Keyword arguments for the LensModel class, dict.
    :param extent: Extent as supplied to matplotlib.pyplot.imshow, tuple.
    :param save_dir: Directory to save image, string.
                     Alternative options:
                        'show' - Show the image instead of saving.
                        'show_debug' - Debug, saves all data shown in the 'show' option to a debug folder.
    :param lens_num: (Optional) Lens serial number, int.
    """
    assert len(data) == len(names), 'Names must match data.'
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, figure=fig)
    subgs = gs[0].subgridspec(1, len(data))
    ax = subgs.subplots()
    image_ax, table_ax = gs.subplots()
    image_ax.axis('off')
    if lens_num is not None: fig.suptitle(f'Lens #{lens_num}')
    for kwargs in kwargs_lens:
        phi, q = el2phiq(kwargs['e1'], kwargs['e2'])
        kwargs.update({'q': q, 'phi': phi})
    table_data = list(map(lambda x: list(x.values()), kwargs_lens))
    col_label = list(kwargs_lens[0].keys())
    row_label = list(range(1, 1 + len(kwargs_lens)))
    for k in range(len(data)):
        if contains(txt=names[k], substr=['convergence', 'kappa']):
            ax[k].imshow(np.log(data[k]), origin='lower', extent=extent)
        else:
            ax[k].imshow(data[k], origin='lower', extent=extent)
        ax[k].title.set_text(names[k])
    table_ax.axis('off')
    table_ax.title.set_text('Lens Parameters')
    tbl = plt.table(cellText=np.around(table_data, decimals=2), colLabels=col_label, rowLabels=row_label,
                    loc='best')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    table_ax.add_table(tbl)
    fig.set_size_inches(3 * len(data), 8)
    if 'show' in save_dir.lower():
        if 'debug' in save_dir.lower():
            base = 'debug'
            save_dir = f'Lens{lens_num if lens_num is not None else ""}/'
            if not isdir('/'.join((base, save_dir))):
                Path(base, save_dir).mkdir(parents=True, exist_ok=True)
            [np.save(file=f'{base}/{save_dir}/{x.removesuffix("(log)").removesuffix(" ")}.npy', arr=y,
                     allow_pickle=False)
             for x, y in zip(names, data)]
            fig.savefig(f'{base}/Lens{lens_num if lens_num is not None else ""}.jpg', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    else:
        fig.savefig(f'{save_dir}/instance{lens_num if lens_num is not None else ""}.jpg', bbox_inches='tight')
        plt.close(fig)


def to_img(data: np.ndarray):
    """
    Converts array to image format.
    :param data: Data to convert, np.ndarray.
    :return: Reshaped array into image format as np.ndarray.
    """
    assert len(data.shape) <= 2, f'np.ndarray with shape={data.shape} is invalid'
    if len(data.shape) == 1:
        data = array2image(data)
    return data.reshape(data.shape + (1,))


def generate_stack(npix, deltapix, stack_size, light_model):
    """
    Generates a stack of lensing simulation instances.

    :param npix: Number of pixels per dimension, int.
    :param deltapix: Pixel resolution (arcsec/pixel), float.
    :param stack_size: Size of stack to generate, int.
    :param light_model: Light source for which to simulate lensing, LightModel instance.
    :return: Stack of inputs and labels for each simulated instance as list of np.ndarray.
    """
    kdata = np.zeros((stack_size, npix, npix, 1))  # Container for kappa data
    imdata = np.zeros((stack_size, npix, npix, 1))  # Container for image data
    for j in range(stack_size):
        kappa, image = generate_instance(npix=npix, deltapix=deltapix, light_model=light_model)
        kdata[j] = kappa
        imdata[j] = image
    return [kdata, imdata]


def generate_instance(npix, deltapix, light_model=None, save_dir=None, instance=None, kwargs_light=None,
                      kwargs_lens=None):
    """
    Generates an instance of lensing simulation - lensed image and lensing mass distribution (convergence).
    The output of this function is dependent on the save_dir keyword.
    By default this function will return the data as a list of np.ndarrays.

    :param npix: Number of pixels per dimension, int.
    :param deltapix: Pixel resolution (arcsec/pixels), float.
    :param light_model: (Optional) Light source, LightModel instance. Defaults to Sersic.
    :param save_dir: (Optional) Defaults to None. See docs of make_image for more options.
    :param instance: (Optional) Simulation instance number,int.
    :param kwargs_light: (Optional) Keyword arguments for a light profile.
    :param kwargs_lens: (Optional) Keyword arguments for a lens profile.
    :return: Lensed image and convergence, respectively, as a list of two np.ndarrays. This output can be changed.
    """
    kwargs_light = {} if kwargs_light is None else kwargs_light
    kwargs_lens = {} if kwargs_lens is None else kwargs_lens
    light_model = light_model or LightModel(['SERSIC'])
    # numeric kwargs
    kwargs_nums = {'supersampling_factor': 1, 'supersampling_convolution': False}
    # PSF
    kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.1, 'pixel_size': deltapix}
    psf = PSF(**kwargs_psf)
    # Pixel Grid
    pixel_grid = grid(npix=npix, deltapix=deltapix, origin_ra=0, origin_dec=0)
    xgrid, ygrid = make_grid(numPix=npix, deltapix=deltapix)
    # Light center
    light_center = (lambda x: x if all(y is not None for y in x) else sample_center(pixel_grid, margin=0.45))(
        list(map(kwargs_light.get, ['center_x', 'center_y'])))
    kwargs_light = kwargs_light or {'amp': 1, 'R_sersic': 1 / 8, 'n_sersic': 3 / 2}
    for key, val in zip(['center_x', 'center_y'], light_center):
        kwargs_light.update({key: val})
    kwargs_light = [kwargs_light]
    # Lens
    if not kwargs_lens:
        lens_model, kwargs_lens = generate_lens(grid_class=pixel_grid)
    else:
        lens_model = LensModel(['SIE'] * len(kwargs_lens))
    # Image Model
    image_model = ImageModel(data_class=pixel_grid, psf_class=psf, lens_model_class=lens_model,
                             source_model_class=light_model,
                             lens_light_model_class=None,
                             kwargs_numerics=kwargs_nums)  # , point_source_class=None)
    # We can add noise to images, see Lenstronomy documentation and examples.
    image = to_img(image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_light,
                                     lens_light_add=False, kwargs_ps=None))
    kappa = to_img(lens_model.kappa(x=xgrid, y=ygrid, kwargs=kwargs_lens))

    # For simulation of images - mainly testing purposes
    if save_dir is None:
        return [kappa, image]
    else:
        # brightness is not implemented in saving but this can be done very quickly by adding a few lines
        # very similar to the commands saving imdata and kdata
        brightness = to_img(light_model.surface_brightness(x=xgrid, y=ygrid, kwargs_list=kwargs_light))
        alpha_x, alpha_y = list(map(to_img, lens_model.alpha(x=xgrid, y=ygrid, kwargs=kwargs_lens)))
        make_image(data=[image, kappa, brightness, alpha_x, alpha_y],
                   names=['Lensed Image', 'Convergence (log)', 'Galaxy w.o Lensing', 'Alpha x', 'Alpha y'],
                   kwargs_lens=kwargs_lens,
                   lens_num=instance,
                   extent=(xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()),
                   save_dir=save_dir)


def save_stack(kdata, imdata, num_train, num_val, stack_size, positions, training_dir):
    """
    Saves a stack to numpy files.
    Will attempt to save to each of the four possible variants:
        Training inputs
        Training labels
        Validation inputs
        Validation labels

    If no validation data is required set num_val = 0.

    :param kdata: Convergence data to save, np.ndarray.
    :param imdata: Image data to save, np.ndarray.
    :param num_train: Number of training instances, int.
    :param num_val: Number of validation instances, int.
    :param stack_size: Stack size, int.
    :param positions: Starting positions in which to write for each file in the following order, tuple:
                        (training input, training label, validation  input, validation label)

    :param training_dir: Save directory, string.
    :return: A tuple of next empty slots of each file in the order prescribed under positions parameter.
    """
    # The following is just a mechanism of separating the training from the validation sets by counting down the
    # number of instances to be generated
    assert len(positions) == 4, f'{positions} is not a valid positions value.'
    train_inp_pos, train_lab_pos, val_inp_pos, val_lab_pos = positions
    if num_train > 0:
        if num_train >= stack_size:
            # save all to training
            # reduce num_train
            train_inp_pos = npy_write(f'{training_dir}/training_input.npy', train_inp_pos, imdata, size=num_train)
            train_lab_pos = npy_write(f'{training_dir}/training_label.npy', train_lab_pos, kdata, size=num_train)
            num_train -= stack_size
        else:
            # save only remaining to training and rest to validation
            # reduce num_train
            # reduce num_val
            train_inp_pos = npy_write(f'{training_dir}/training_input.npy', train_inp_pos, imdata[:num_train],
                                      size=num_train)
            train_lab_pos = npy_write(f'{training_dir}/training_label.npy', train_lab_pos, kdata[:num_train],
                                      size=num_train)
            if num_val >= stack_size - num_train:
                val_inp_pos = npy_write(f'{training_dir}/validation_input.npy', val_inp_pos, imdata[num_train:],
                                        size=num_val)
                val_lab_pos = npy_write(f'{training_dir}/validation_label.npy', val_lab_pos, kdata[num_train:],
                                        size=num_val)
                num_val -= (stack_size - num_train)
            else:
                print('Warning! Too many instances generated.')
            num_train -= num_train
    elif num_val > 0:
        if num_val >= stack_size:
            # save all to validation
            # reduce num_val
            val_inp_pos = npy_write(f'{training_dir}/validation_input.npy', val_inp_pos, imdata, size=num_val)
            val_lab_pos = npy_write(f'{training_dir}/validation_label.npy', val_lab_pos, kdata, size=num_val)
            num_val -= stack_size
        else:
            print('Warning! Too many instances generated.')
    elif num_val < 0 or num_train < 0:
        raise ValueError(f'Negative num: num_train={num_train},num_val={num_val}')
    return [(train_inp_pos, train_lab_pos, val_inp_pos, val_lab_pos), num_train, num_val]


def generate_training(npix, deltapix, stacks, stack_size, val_split=VAL_SPLIT, **kwargs):
    """
    Generates and saves training data to user specified directory.
    Separates the generation into stacks.
    Option 1 of the main menu.

    :param npix: Number of pixels per dimension, int.
    :param deltapix: Pixel resolution (arcsec/pixels), float.
    :param stacks: Number of stacks to generate, int.
    :param stack_size: Size of each stack, int.
    :param val_split: (Optional) Validation split fraction, float. Defaults to VAL_SPLIT global variable.
    :return: training directory as string.
    """

    # stack_size is also the size of the np array initialized to store the images - has memory implications.
    # val_split = 0.1
    training_dir = get_dir_gui(title='Select folder to save training data')
    num_train, num_val = np.around(stack_size * stacks * np.array([1 - val_split, val_split])).astype('int')
    if num_train + num_val == stacks * stack_size + 1:
        num_train -= 1
    assert num_train + num_val == stacks * stack_size, 'Number of training and validation instances does not match stack and stack_size.'
    positions = (0, 0, 0, 0)
    light_model = LightModel(light_model_list=['SERSIC'])
    for i in range(stacks):
        kdata, imdata = generate_stack(npix=npix, deltapix=deltapix, stack_size=stack_size, light_model=light_model)

        positions, num_train, num_val = save_stack(kdata=kdata, imdata=imdata, num_train=num_train, num_val=num_val,
                                                   stack_size=stack_size, positions=positions,
                                                   training_dir=training_dir)
    return training_dir


def generate_image(npix, deltapix, stacks, stack_size=1, action='show', **kwargs):
    """
    Generates a single image.
    Options 2 and 3 of the main menu.
    Parameters 'stack' and 'stack_size' exist as artifacts of the method.
    They are non-independent in this case and it is sufficient to just supply 'stacks'.
    Ultimately their values are multiplied to get the number of images required.

    :param npix: Number of pixels per dimension, int.
    :param deltapix: Pixel resolution (arcsec/pixels), float.
    :param stacks: Number of stacks to generate, int.
    :param stack_size: (Optional) Size of each stack, int. Defaults to 1.
    :param action: (Optional) Defaults to 'show'.
                   Alternatives:
                    'save_img' - Saves the image and will prompt for a directory.

                    See further options under make_image documentation.
    """
    light_model = LightModel(light_model_list=['SERSIC'])
    if action == 'save_img':
        img_dir = get_dir_gui(title='Image save location')
    else:
        img_dir = action
    stack_size *= stacks
    for i in range(stack_size):
        generate_instance(npix=npix, deltapix=deltapix, light_model=light_model, save_dir=img_dir, instance=i + 1)


def simulation(npix, deltapix, stacks, stack_size, action: str, val_split=VAL_SPLIT):
    """
    Precursor function to all user directed simulation objectives.
    Will initiate the user menu and follow the workflow of the data generation process.
    Functions from this workflow can be used independently along with appropriate input arguments.

    :param npix:
    :param deltapix:
    :param stacks:
    :param stack_size:
    :param action:
    :param val_split:
    :return:
    """
    options = {'save': generate_training, 'save_img': generate_image, 'show': generate_image}
    func = options.get(action.lower(), None)
    if func is None:
        raise ValueError(f'{action} is not a recognized action.')
    else:
        return func(npix=npix, deltapix=deltapix, stacks=stacks, stack_size=stack_size, val_split=val_split,
                    action=action.lower())


def main():
    # Constant constructs #
    # The following constructs are shared by all generated lensing instances
    simulation(npix=150, deltapix=0.1, stacks=100, stack_size=1000, action='save')


if __name__ == '__main__':
    main()
