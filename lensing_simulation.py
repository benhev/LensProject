# from lenstronomy.Util.param_util import phi_q2_ellipticity as qphi2el
# import matplotlib.pyplot as plt
from lenstronomy.Util.util import make_grid, array2image
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
import numpy as np
import lzma
import pickle


def dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def distance_check(lens_centers, light_center, lens_thetas):
    if len(lens_centers) != len(lens_thetas):
        raise TypeError('Length of Einstein Radii and lens centers lists not of same length!')
    for center, theta in zip(lens_centers, lens_thetas):
        if dist(center, light_center) < theta:
            return True
    return False


def grid(dpix, npix, origin_ra, origin_dec):
    p2a_transform = np.array([[1, 0], [0, 1]]) * dpix
    kwargs = {'image_data': np.zeros((npix, npix)), 'ra_at_xy_0': origin_ra,
              'dec_at_xy_0': origin_dec, 'transform_pix2angle': p2a_transform}
    return ImageData(**kwargs)


def generate_lens(grid_class):
    num_of_lenses = 1  # np.random.randint(1, 6)
    grid_len = len(grid_class.pixel_coordinates[0])
    margin = np.floor(grid_len / 10)
    kwargs = []
    centers = []
    thetas = []
    pars = []
    for _ in range(num_of_lenses):
        e1, e2 = 0, 0  # np.random.uniform(-10, 10, 2)
        theta = 1  # np.random.uniform(1, 3)
        cx, cy = 0, 0  # np.random.randint(margin, grid_len + 1 - margin, 2)
        cx, cy = grid_class.map_pix2coord(x=cx, y=cy)
        temp_kwargs = {'theta_E': theta, 'e1': e1, 'e2': e2, 'center_x': cx, 'center_y': cy}
        kwargs.append(temp_kwargs)
        centers.append((cx, cy))
        thetas.append(theta)
        pars.append((e1, e2))
    print('Number of lenses:', num_of_lenses)
    print('Centers:', centers)
    print('Ellipticity:', pars)
    print('Einstein Angles:', thetas)
    model = LensModel(lens_model_list=['SIE'] * num_of_lenses)
    return [model, kwargs, centers, thetas]


def light(grid_class, lens_centers, lens_thetas):
    r, n = 1 / 2, 3 / 2
    grid_len = len(grid_class.pixel_coordinates[0])
    margin = np.floor(grid_len / 10)
    # print('Grid Length:', grid_len)
    cond = False
    # iterator = 0
    while not cond:
        cx, cy = 0, 0  # np.random.randint(margin, grid_len + 1 - margin, 2)
        cx, cy = grid_class.map_pix2coord(x=cx, y=cy)
        cond = distance_check(lens_centers=lens_centers, lens_thetas=lens_thetas, light_center=(cx, cy))
        # print('Iteration:', iterator)
        # iterator += 1
    kwargs = [{'amp': 1, 'R_sersic': r, 'n_sersic': n, 'center_x': cx, 'center_y': cy}]
    model = LightModel(light_model_list=['SERSIC'])
    # print('Light Center:', (cx, cy))
    return [model, kwargs]


# Constant constructs #
# The following constructs up to the loop environment are shared by all generated lensing instances
deltapix = 0.05  # size of pixel in angular coordinates
npix = 100  # image shape = (npix, npix, 1)
save_dir = 'Training Set'
# Generates stacks bunches of stack_size images.
# stack_size is also the size of the np array initialized to store the images - has memory implications.
# In a future update these numbers won't make a difference as all data will be appended to one numpy file.
stack_size = 1  # 0000
stacks = 1  # 0
kwargs_nums = {'supersampling_factor': 1, 'supersampling_convolution': False}  # numeric kwargs
# PSF
kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.1, 'pixel_size': deltapix}
psf = PSF(**kwargs_psf)
kernel = psf.kernel_point_source
# Pixel Grid
pgrid = grid(dpix=deltapix, npix=npix, origin_ra=0, origin_dec=0)
xgrid, ygrid = make_grid(numPix=npix, deltapix=deltapix)

# We can add noise to images, see Lenstronomy documentation and examples.

for i in range(stacks):
    data = []
    # kdata = np.zeros((stack_size, npix, npix, 1))
    # imdata = np.zeros((stack_size, npix, npix, 1))
    for _ in range(stack_size):
        # Lens
        lensModel, kwargs_lens, centers, thetas = generate_lens(grid_class=pgrid)

        lightModel, kwargs_light = light(grid_class=pgrid, lens_centers=centers, lens_thetas=thetas)
        # Image Model
        imageModel = ImageModel(data_class=pgrid, psf_class=psf, lens_model_class=lensModel,
                                source_model_class=lightModel,
                                lens_light_model_class=None, kwargs_numerics=kwargs_nums)  # , point_source_class=None)
        image = imageModel.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_light,
                                 lens_light_add=False, kwargs_ps=None)#, point_source_add=False)
        kappa = array2image(lensModel.kappa(x=xgrid, y=ygrid, kwargs=kwargs_lens))
        # image = image.reshape(image.shape[0], image.shape[1], 1)
        # kappa = kappa.reshape(kappa.shape[0], kappa.shape[1], 1)
        # kdata[_] = kappa
        # imdata[_] = image
        brightness = array2image(lightModel.surface_brightness(x=xgrid, y=ygrid, kwargs_list=kwargs_light))
        #
        # ax[0, _].matshow(image, origin='lower')
        # ax[1, _].matshow(brightness, origin='lower')
        # ax[2, _].matshow(np.log10(kappa), origin='lower')
        data = [image, kappa, brightness]
        names = ['image', 'kappa', 'brightness']
    fig, ax = plt.subplots(1, 3, sharey='all')
    for i in range(3):
        # if i == 1:
        #     ax[i].imshow(np.log(data[i]))
        # else:
        ax[i].matshow(data[i])
        ax[i].title.set_text(names[i])

    plt.show()
    # data = [imdata, kdata]
    # In a future update this should change to be saved as a numpy file.
    # Numpy files load and save faster, while taking slightly more storage space.
    # filename = f'{save_dir}/lens_set_{str(i + 1)}.xz'
    # with lzma.open(filename, mode='xb') as file:
    #     pickle.dump(data, file)
    # In a future update a validation file can be separated from the training set at this stage
