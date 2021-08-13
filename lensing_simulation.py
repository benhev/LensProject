# # we can also add noise #
# import lenstronomy.Util.image_util as image_util
# exp_time = 100  # exposure time to quantify the Poisson noise level
# background_rms = 0.1  # background rms value
# poisson = image_util.add_poisson(image, exp_time=exp_time)
# bkg = image_util.add_background(image, sigma_bkd=background_rms)
# image_noisy = image + bkg + poisson
#
# f, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)
# axes[0].matshow(np.log10(image), origin='lower')
# axes[1].matshow(np.log10(image_noisy), origin='lower')
# f.tight_layout()
# plt.show()

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
import numpy as np
# from lenstronomy.Util.param_util import phi_q2_ellipticity as qphi2el
import matplotlib.pyplot as plt
import lzma
import pickle


def dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def distance_check(lens_centers, light_center, lens_thetas):
    if len(lens_centers) != len(lens_thetas):
        raise TypeError('Length of Einstein Radii and lens centers lists not of same length!')
    for center, theta in list(zip(lens_centers, lens_thetas)):
        if dist(center, light_center) < 2 * theta:
            return True
    return False


def grid(dpix, npix, origin_ra, origin_dec):
    p2a_transform = np.array([[1, 0], [0, 1]]) * dpix
    kwargs = {'nx': npix, 'ny': npix, 'ra_at_xy_0': origin_ra,
              'dec_at_xy_0': origin_dec, 'transform_pix2angle': p2a_transform}
    return PixelGrid(**kwargs)


def generate_lens(grid_class):
    num_of_lenses = np.random.randint(1, 6)
    grid_len = len(grid_class.pixel_coordinates[0])
    margin = np.floor(grid_len / 10)
    kwargs = []
    centers = []
    thetas = []
    pars = []
    for _ in range(num_of_lenses):
        e1, e2 = np.random.uniform(-0.5, 0.5, 2)
        theta = np.random.uniform(1, 5)
        cx, cy = np.random.randint(margin, grid_len + 1 - margin, 2)
        cx, cy = grid_class.map_pix2coord(x=cx, y=cy)
        temp_kwargs = {'theta_E': theta, 'e1': e1, 'e2': e2, 'center_x': cx, 'center_y': cy}
        kwargs.append(temp_kwargs)
        centers.append((cx, cy))
        thetas.append(theta)
        pars.append((e1, e2))
    # print('Number of lenses:', num_of_lenses)
    # print('Centers:', centers)
    # print('Ellipticity:', pars)
    # print('Einstein Angles:', thetas)
    model = LensModel(lens_model_list=['SIE'] * num_of_lenses)
    return [model, kwargs, centers, thetas]


def light(grid_class, lens_centers, lens_thetas):
    r, n = 5, 1
    grid_len = len(grid_class.pixel_coordinates[0])
    margin = np.floor(grid_len / 10)
    # print('Grid Length:', grid_len)
    cond = False
    # iterator = 0
    while not cond:
        cx, cy = np.random.randint(margin, grid_len + 1 - margin, 2)
        cx, cy = grid_class.map_pix2coord(x=cx, y=cy)
        cond = distance_check(lens_centers=lens_centers, lens_thetas=lens_thetas, light_center=(cx, cy))
        # print('Iteration:', iterator)
        # iterator += 1
    kwargs = [{'amp': 1, 'R_sersic': r, 'n_sersic': n, 'center_x': cx, 'center_y': cy}]
    model = LightModel(light_model_list=['SERSIC'])
    # print('Light Center:', (cx, cy))
    return [model, kwargs]


def file_read(directory):
    with lzma.open(directory, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


# Constant constructs #
# The following constructs up to the loop environment are shared by all generated lensing instances
deltapix = 0.05  # pixel resolution
npix = 100
kwargs_nums = {'supersampling_factor': 1, 'supersampling_convolution': False}  # numeric kwargs
# PSF
kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.1, 'pixel_size': deltapix}
psf = PSF(**kwargs_psf)
kernel = psf.kernel_point_source
# Pixel Grid
pixelGrid = grid(dpix=deltapix, npix=npix, origin_ra=0, origin_dec=0)
xgrid, ygrid = pixelGrid.pixel_coordinates

# f, ax = plt.subplots(3, n, figsize=(8, 8))
# with open('lens_test1.txt', mode='a') as file:.
stack_size = 10000
for i in range(10):
    data=[]
    kdata = np.zeros((stack_size, npix, npix, 1))
    imdata = np.zeros((stack_size, npix, npix, 1))
    for _ in range(stack_size):
        # Lens
        lensModel, kwargs_lens, centers, thetas = generate_lens(grid_class=pixelGrid)
        # Source
        lightModel, kwargs_light = light(grid_class=pixelGrid, lens_centers=centers, lens_thetas=thetas)
        # Image Model
        imageModel = ImageModel(data_class=pixelGrid, psf_class=psf, lens_model_class=lensModel,
                                source_model_class=lightModel,
                                lens_light_model_class=None, point_source_class=None, kwargs_numerics=kwargs_nums)
        image = imageModel.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_light, point_source_add=False,
                                 lens_light_add=False)
        kappa = lensModel.kappa(x=xgrid, y=ygrid, kwargs=kwargs_lens)
        image = image.reshape(image.shape[0], image.shape[1], 1)
        kappa = kappa.reshape(kappa.shape[0], kappa.shape[1], 1)
        kdata[_] = kappa
        imdata[_] = image
        # brightness = lightModel.surface_brightness(x=xgrid, y=ygrid, kwargs_list=kwargs_light)
        #
        # ax[0, _].matshow(image, origin='lower')
        # ax[1, _].matshow(brightness, origin='lower')
        # ax[2, _].matshow(np.log10(kappa), origin='lower')

    data = [imdata, kdata]
    filename = ''.join(['Training Set/lens_set_', str(i + 1), '.xz'])
    with lzma.open(filename, mode='xb') as file:
        pickle.dump(data, file)

# # Lens
# lensModel, kwargs_lens, centers, thetas = generate_lens(grid_class=pixelGrid)
# # Source
# lightModel, kwargs_light = light(grid_class=pixelGrid, lens_centers=centers, lens_thetas=thetas)
# # Image Model
# imageModel = ImageModel(data_class=pixelGrid, psf_class=psf, lens_model_class=lensModel,
#                         source_model_class=lightModel,
#                         lens_light_model_class=None, point_source_class=None, kwargs_numerics=kwargs_nums)
# image = imageModel.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_light, point_source_add=False,
#                          lens_light_add=False)
# kappa = lensModel.kappa(x=xgrid, y=ygrid, kwargs=kwargs_lens)
#
#
# brightness = lightModel.surface_brightness(x=xgrid, y=ygrid, kwargs_list=kwargs_light)
# f, ax = plt.subplots(1, 3, figsize=(8, 8))
# ax[0].matshow(np.log10(kappa), origin='lower')
# ax[1].matshow(image, origin='lower')
# ax[2].matshow(brightness, origin='lower')
#
# f.tight_layout()
# plt.show()
