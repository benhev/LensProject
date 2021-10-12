from lenstronomy.Util.param_util import phi_q2_ellipticity as phiq2el
from lenstronomy.Util.param_util import ellipticity2phi_q as el2phiq
# from lenstronomy.Data.imaging_data import ImageData
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

from LensCNN import get_dir

import winsound

FREQ = 1000  # Set Frequency To 2500 Hertz
DUR = 500  # Set Duration To 1000 ms == 1 second

MARGIN_FRACTION = 0.4
VAL_SPLIT = 0.1


def dist_check(lens_center, prev_centers, rng):
    prev_centers = np.array(prev_centers)

    # finish distance checking so two lenses are not too close together
    for center in prev_centers:
        if np.linalg.norm(center - lens_center) < rng:
            return True
    return False  # True if len(prev_centers) == 0 else False



def npy_write(filename: str, start_row, arr, size=None):
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
        start_byte = np.int64(start_row * row_size * dtype.itemsize)
        file.seek(start_byte, 1)
        arr.tofile(file)
    return start_row + num_rows


def grid(npix, deltapix, origin_ra, origin_dec):
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, p2a_transform, _ = make_grid_with_coordtransform(numPix=npix,
                                                                                          deltapix=deltapix,
                                                                                          center_ra=origin_ra,
                                                                                          center_dec=origin_dec,
                                                                                          subgrid_res=1,
                                                                                          inverse=False)
    kwargs = {'nx': npix, 'ny': npix, 'ra_at_xy_0': ra_at_xy_0,
              'dec_at_xy_0': dec_at_xy_0, 'transform_pix2angle': p2a_transform}
    return PixelGrid(**kwargs)


def generate_lens(grid_class, light_center):
    num_of_lenses = np.random.randint(1, 5)
    # light_center = np.array(light_center)
    kwargs = []
    centers = []
    for _ in range(num_of_lenses):
        e1, e2 = phiq2el(np.random.uniform(0, np.pi), np.random.uniform(0.7, 1.0))
        theta = np.random.normal(loc=1, scale=0.01)
        # center = np.array([np.inf, np.inf])
        # while np.linalg.norm(light_center - center) > 3 * theta or dist_check(lens_center=center,
        #                                                                       prev_centers=centers, rng=theta):
        center = sample_center(grid_class)
        centers.append(center)
        temp_kwargs = {'center_x': center[0], 'center_y': center[1], 'theta_E': theta, 'e1': e1, 'e2': e2}
        kwargs.append(temp_kwargs)

    return [LensModel(lens_model_list=['SIE'] * num_of_lenses), kwargs]


def contains(txt: str, substr: list):
    for line in substr:
        if not isinstance(line, str):
            raise ValueError(f'{substr} is not a list of strings.')
        if line.lower() in txt.lower():
            return True
    return False


def sample_center(grid_class, margin=MARGIN_FRACTION):
    assert margin <= 1 / 2, 'MARGIN_FRACTION must be <=1/2.'
    xmin, ymin = (1 - 2 * margin) * np.array(list(map(np.min, grid_class.pixel_coordinates)))
    xmax, ymax = (1 - 2 * margin) * np.array(list(map(np.max, grid_class.pixel_coordinates)))
    # grid_len = len(grid_class.pixel_coordinates[0])
    # margin = np.floor(MARGIN_FRACTION * grid_len)
    cx, cy = list(map(lambda x: np.random.uniform(*x), [(xmin, xmax), (ymin, ymax)]))
    # cx, cy = np.around(grid_class.map_pix2coord(x=cx, y=cy), decimals=1)
    return np.array([cx, cy])


def make_image(data, names, kwargs_lens, extent, save_dir, lens_num=None):
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
    # fig_size = fig.get_size_inches()
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
    assert len(data.shape) <= 2, f'np.ndarray with shape={data.shape} is invalid'
    if len(data.shape) == 1:
        data = array2image(data)
    return data.reshape(data.shape + (1,))


def generate_stack(npix, deltapix, stack_size, light_model):
    kdata = np.zeros((stack_size, npix, npix, 1))  # Container for kappa data
    imdata = np.zeros((stack_size, npix, npix, 1))  # Container for image data
    for j in range(stack_size):
        kappa, image = generate_instance(npix=npix, deltapix=deltapix, light_model=light_model)
        kdata[j] = kappa
        imdata[j] = image
    return [kdata, imdata]


def generate_instance(npix, deltapix, light_model=None, save_dir=None, instance=None, kwargs_light=None,
                      kwargs_lens=None):
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
        lens_model, kwargs_lens = generate_lens(grid_class=pixel_grid, light_center=light_center)
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
    #
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
    # Generates {stacks} bunches of {stack_size images}.
    # stack_size is also the size of the np array initialized to store the images - has memory implications.
    # val_split = 0.1
    training_dir = get_dir('training', new=True)
    num_train, num_val = np.around(stack_size * stacks * np.array([1 - val_split, val_split])).astype('int')
    if num_train + num_val == stacks * stack_size + 1:
        num_train -= 1
    assert num_train + num_val == stacks * stack_size, 'Number of training and validation instances does not match stack and stack_size.'
    positions = (0, 0, 0, 0)
    light_model = LightModel(light_model_list=['SERSIC'])
    for i in range(stacks):
        kdata, imdata = generate_stack(npix=npix, deltapix=deltapix, stack_size=stack_size, light_model=light_model)
        # if not (num_train % 10000):
        #     winsound.Beep(FREQ, DUR)
        #     print(True)

        positions, num_train, num_val = save_stack(kdata=kdata, imdata=imdata, num_train=num_train, num_val=num_val,
                                                   stack_size=stack_size, positions=positions,
                                                   training_dir=training_dir)


def generate_image(npix, deltapix, stacks, stack_size, action='show', **kwargs):
    light_model = LightModel(light_model_list=['SERSIC'])
    if action == 'save_img':
        img_dir = get_dir('image', new=False)
    else:
        img_dir = action
    stack_size *= stacks
    for i in range(stack_size):
        generate_instance(npix=npix, deltapix=deltapix, light_model=light_model, save_dir=img_dir, instance=i + 1)


def simulation(npix, deltapix, stacks, stack_size, action: str):
    options = {'save': generate_training, 'save_img': generate_image, 'show': generate_image}
    func = options.get(action.lower(), None)
    if func is None:
        raise ValueError(f'{action} is not a recognized action.')
    else:
        func(npix=npix, deltapix=deltapix, stacks=stacks, stack_size=stack_size, val_split=VAL_SPLIT,
             action=action.lower())


def main():
    # Constant constructs #
    # The following constructs are shared by all generated lensing instances
    simulation(npix=150, deltapix=0.1, stacks=100, stack_size=1000, action='save')


if __name__ == '__main__':
    main()
