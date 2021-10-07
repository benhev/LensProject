# from lenstronomy.Util.param_util import phi_q2_ellipticity as qphi2el
# import matplotlib.pyplot as plt
from lenstronomy.Util.util import make_grid, array2image, make_grid_with_coordtransform
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
# from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
import numpy as np
from os.path import isfile, isdir, dirname
from pathlib import Path


def npy_write(filename: str, start_row, arr, size=None):
    assert start_row >= 0 and isinstance(arr, np.ndarray)
    num_rows = len(arr) if len(arr.shape) > 1 else 1
    if not isfile(filename):
        assert filename.endswith('.npy'), 'File name must end with .npy.'
        if size is not None and num_rows <= size:
            if not isdir(dirname(filename)):
                Path(dirname(filename)).mkdir()
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
        start_byte = start_row * row_size * dtype.itemsize
        file.seek(start_byte, 1)
        arr.tofile(file)
        return start_row + num_rows


def grid(dpix, npix, origin_ra, origin_dec):
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, p2a_transform, _ = make_grid_with_coordtransform(numPix=npix,
                                                                                          deltapix=dpix,
                                                                                          center_ra=origin_ra,
                                                                                          center_dec=origin_dec,
                                                                                          subgrid_res=1,
                                                                                          inverse=False)
    kwargs = {'nx': npix, 'ny': npix, 'ra_at_xy_0': ra_at_xy_0,
              'dec_at_xy_0': dec_at_xy_0, 'transform_pix2angle': p2a_transform}
    return PixelGrid(**kwargs)


def generate_lens(grid_class, light_center):
    num_of_lenses = np.random.randint(2, 5)
    grid_len = len(grid_class.pixel_coordinates[0])
    margin = np.floor(grid_len / 5)
    light_center = np.array(light_center)
    kwargs = []
    # centers = []
    # print(f"Light Center {light_center}")
    for _ in range(num_of_lenses):
        e1, e2 = np.random.uniform(-0.5, 0.5, 2)
        # print(f'ellipticity #{_ + 1}={(e1, e2)}')
        theta = np.random.normal(loc=1, scale=0.01)
        center = np.array([np.inf, np.inf])
        while np.linalg.norm(light_center - center) > 3 * theta:
            center = np.random.randint(margin, grid_len + 1 - margin, 2)
            center = np.around(grid_class.map_pix2coord(x=center[0], y=center[1]), decimals=1)
        # print(f'Center #{_ + 1}={center}')
        # centers.append(center)
        temp_kwargs = {'theta_E': theta, 'e1': e1, 'e2': e2, 'center_x': center[0], 'center_y': center[1]}
        kwargs.append(temp_kwargs)
    model = LensModel(lens_model_list=['SIE'] * num_of_lenses)
    return [model, kwargs]


def light(grid_class):
    r, n = 1. / 2, 3 / 2
    grid_len = len(grid_class.pixel_coordinates[0])
    margin = np.floor(grid_len / 5)
    cx, cy = np.random.randint(margin, grid_len + 1 - margin, 2)
    cx, cy = np.around(grid_class.map_pix2coord(x=cx, y=cy), decimals=2)
    kwargs = [{'amp': 1, 'R_sersic': r, 'n_sersic': n, 'center_x': cx, 'center_y': cy}]
    model = LightModel(light_model_list=['SERSIC'])
    return [model, kwargs, np.array([cx, cy])]


def main():
    # Constant constructs #
    # The following constructs up to the "for" block are shared by all generated lensing instances
    deltapix = 0.05  # size of pixel in angular coordinates
    npix = 150  # image shape = (npix, npix, 1)
    save_dir = input('Input save directory:')
    while isdir(save_dir):
        print('Directory exists!')
        save_dir = input('Input save directory:')
    # Generates stacks bunches of stack_size images.
    # stack_size is also the size of the np array initialized to store the images - has memory implications.
    # In a future update these numbers won't make a difference as all data will be appended to one numpy file.
    stack_size = 11  # 0000
    stacks = 13  # 0
    val_split = 0.1
    kwargs_nums = {'supersampling_factor': 1, 'supersampling_convolution': False}  # numeric kwargs
    # PSF
    kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.1, 'pixel_size': deltapix}
    psf = PSF(**kwargs_psf)
    # kernel = psf.kernel_point_source
    # Pixel Grid
    pixel_grid = grid(dpix=deltapix, npix=npix, origin_ra=0, origin_dec=0)
    xgrid, ygrid = make_grid(numPix=npix, deltapix=deltapix)

    # We can add noise to images, see Lenstronomy documentation and examples.
    num_train, num_val = np.around(stack_size * stacks * np.array([1 - val_split, val_split])).astype('int')
    if num_train + num_val == stacks * stack_size + 1:
        num_train -= 1
    assert num_train + num_val == stacks * stack_size, 'Number of training and validation instances does not match stack and stack_size.'
    train_inp_pos, train_lab_pos, val_inp_pos, val_lab_pos = 0, 0, 0, 0
    for i in range(stacks):
        kdata = np.zeros((stack_size, npix, npix, 1))  # Container for kappa data
        imdata = np.zeros((stack_size, npix, npix, 1))  # Container for image data
        for _ in range(stack_size):
            # Light
            light_model, kwargs_light, light_center = light(grid_class=pixel_grid)
            # Lens
            lens_model, kwargs_lens = generate_lens(grid_class=pixel_grid, light_center=light_center)
            # Image Model
            image_model = ImageModel(data_class=pixel_grid, psf_class=psf, lens_model_class=lens_model,
                                     source_model_class=light_model,
                                     lens_light_model_class=None,
                                     kwargs_numerics=kwargs_nums)  # , point_source_class=None)
            image = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_light,
                                      lens_light_add=False, kwargs_ps=None)  # , point_source_add=False)
            kappa = array2image(lens_model.kappa(x=xgrid, y=ygrid, kwargs=kwargs_lens))
            image = image.reshape(image.shape[0], image.shape[1], 1)  # Reshape to fit the expected input of tensorflow
            kappa = kappa.reshape(kappa.shape[0], kappa.shape[1], 1)
            kdata[_] = kappa
            imdata[_] = image
            # if we want we can add brightness plots
            # brightness is not implemented in saving but this can be done very quickly by adding a few lines
            # very similar to the commands saving imdata and kdata
            # brightness = array2image(light_model.surface_brightness(x=xgrid, y=ygrid, kwargs_list=kwargs_light))

            # For simulation of images - mainly testing purposes
            # data = [image, kappa, brightness]
            # names = ['image', 'kappa', 'galaxy w/o lensing']
            # fig, ax = plt.subplots(1, 3)
            # for j in range(3):
            #     if j == 1:
            #         ax[j].imshow(np.log(data[j]))
            #     else:
            #         ax[j].matshow(data[j], extent=(xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()))
            #     ax[j].title.set_text(names[j])
            # fig_size = fig.get_size_inches()
            # fig.set_size_inches(fig_size[0], fig_size[1] / 2)
            # fig.savefig(f'example {_ + 1}.jpg')
            # plt.show()
        # The following is just a mechanism of separating the training from the validation sets by counting down the
        # number of instances to be generated
        if num_train > 0:
            if num_train >= stack_size:
                # save all to training
                # reduce num_train
                train_inp_pos = npy_write(f'{save_dir}/training_input.npy', train_inp_pos, imdata, size=num_train)
                train_lab_pos = npy_write(f'{save_dir}/training_label.npy', train_lab_pos, kdata, size=num_train)
                num_train -= stack_size
            else:
                # save only remaining to training and rest to validation
                # reduce num_train
                # reduce num_val
                train_inp_pos = npy_write(f'{save_dir}/training_input.npy', train_inp_pos, imdata[:num_train],
                                          size=num_train)
                train_lab_pos = npy_write(f'{save_dir}/training_label.npy', train_lab_pos, kdata[:num_train],
                                          size=num_train)
                if num_val >= stack_size - num_train:
                    val_inp_pos = npy_write(f'{save_dir}/validation_input.npy', val_inp_pos, imdata[num_train:],
                                            size=num_val)
                    val_lab_pos = npy_write(f'{save_dir}/validation_label.npy', val_lab_pos, kdata[num_train:],
                                            size=num_val)
                    num_val -= stack_size - num_train
                else:
                    print('Warning! Too many instances generated.')
                num_train -= num_train

        elif num_val > 0:
            if num_val >= stack_size:
                # save all to validation
                # reduce num_val
                val_inp_pos = npy_write(f'{save_dir}/validation_input.npy', val_inp_pos, imdata, size=num_val)
                val_lab_pos = npy_write(f'{save_dir}/validation_label.npy', val_lab_pos, kdata, size=num_val)
                num_val -= stack_size
            else:
                print('Warning! Too many instances generated.')
        elif num_val < 0:
            raise ValueError(f'Negative num_val: num_val={num_val}')


if __name__ == '__main__':
    main()
