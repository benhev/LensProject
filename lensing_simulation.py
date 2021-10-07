# from lenstronomy.Util.param_util import phi_q2_ellipticity as qphi2el
# import matplotlib.pyplot as plt
from lenstronomy.Util import util
from lenstronomy.Util.util import make_grid, array2image
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
# from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
import numpy as np


# import lzma
# import pickle
def dist_check(lens_center, prev_centers, range):
    prev_centers = np.array(prev_centers)
    # finish distance checking so two lenses are not too close together
    for center in prev_centers:
        if np.linalg.norm(center - lens_center) < range:
            return True
    return False


def grid(dpix, npix, origin_ra, origin_dec):
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, p2a_transform, _ = util.make_grid_with_coordtransform(numPix=npix,
                                                                                               deltapix=dpix,
                                                                                               center_ra=origin_ra,
                                                                                               center_dec=origin_dec,
                                                                                               subgrid_res=1,
                                                                                               inverse=False)
    kwargs = {'nx': npix, 'ny': npix, 'ra_at_xy_0': ra_at_xy_0,
              'dec_at_xy_0': dec_at_xy_0, 'transform_pix2angle': p2a_transform}
    return PixelGrid(**kwargs)


def generate_lens(grid_class, light_center):
    num_of_lenses = np.random.randint(1, 4)
    grid_len = len(grid_class.pixel_coordinates[0])
    margin = np.floor(grid_len / 5)
    light_center = np.array(light_center)
    kwargs = []
    centers = []
    for _ in range(num_of_lenses):
        e1, e2 = np.random.uniform(-1 / 2, 1 / 2, 2)
        theta = np.random.normal(loc=1, scale=0.01)
        center = np.array([np.inf, np.inf])
        while np.linalg.norm(light_center - center) > 3 * theta or dist_check(lens_center=center, prev_centers=centers,
                                                                               range=theta):
            center = np.random.randint(margin, grid_len + 1 - margin, 2)
            center = np.around(grid_class.map_pix2coord(x=center[0], y=center[1]), decimals=1)
        print(f"Light Center {light_center}")
        print(f"Center #{_ + 1}={center}")
        centers.append(center)
        temp_kwargs = {'theta_E': theta, 'e1': e1, 'e2': e2, 'center_x': center[0], 'center_y': center[1]}
        kwargs.append(temp_kwargs)
    model = LensModel(lens_model_list=['SIE'] * num_of_lenses)
    return [model, kwargs]


def light(grid_class):
    r, n = 1. / 2, 3 / 2
    grid_len = len(grid_class.pixel_coordinates[0])
    margin = np.floor(grid_len / 5)
    # print('Grid Length:', grid_len)
    # iterator = 0
    cx, cy = np.random.randint(margin, grid_len + 1 - margin, 2)
    cx, cy = np.around(grid_class.map_pix2coord(x=cx, y=cy), decimals=2)
    # print('Iteration:', iterator)
    # iterator += 1
    kwargs = [{'amp': 1, 'R_sersic': r, 'n_sersic': n, 'center_x': cx, 'center_y': cy}]
    model = LightModel(light_model_list=['SERSIC'])
    # print('Light Center:', (cx, cy))
    return [model, kwargs, np.array([cx, cy])]


def main():
    # Constant constructs #
    # The following constructs up to the "for" block are shared by all generated lensing instances
    deltapix = 0.05  # size of pixel in angular coordinates
    npix = 300  # image shape = (npix, npix, 1)
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
    # kernel = psf.kernel_point_source
    # Pixel Grid
    pixel_grid = grid(dpix=deltapix, npix=npix, origin_ra=0, origin_dec=0)
    xgrid, ygrid = util.make_grid(numPix=npix, deltapix=deltapix)

    # We can add noise to images, see Lenstronomy documentation and examples.

    for i in range(stacks):
        data = []
        # kdata = np.zeros((stack_size, npix, npix, 1))
        # imdata = np.zeros((stack_size, npix, npix, 1))
        for _ in range(stack_size):
            # light model should really be first, needs to be changed in code from distance checking the light source
            # to distance checking the lens centers
            # Light
            light_model, kwargs_light, center = light(grid_class=pixel_grid)
            # Lens
            lens_model, kwargs_lens = generate_lens(grid_class=pixel_grid, light_center=center)
            # Image Model
            image_model = ImageModel(data_class=pixel_grid, psf_class=psf, lens_model_class=lens_model,
                                     source_model_class=light_model,
                                     lens_light_model_class=None,
                                     kwargs_numerics=kwargs_nums)  # , point_source_class=None)
            image = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_light,
                                      lens_light_add=False, kwargs_ps=None)  # , point_source_add=False)
            kappa = array2image(lens_model.kappa(x=xgrid, y=ygrid, kwargs=kwargs_lens))
            # image = image.reshape(image.shape[0], image.shape[1], 1)
            # kappa = kappa.reshape(kappa.shape[0], kappa.shape[1], 1)
            # kdata[_] = kappa
            # imdata[_] = image

            brightness = array2image(light_model.surface_brightness(x=xgrid, y=ygrid, kwargs_list=kwargs_light))
            data = [image, kappa, brightness]
            names = ['image', 'kappa', 'surface brightness w/o lensing']
            fig, ax = plt.subplots(1, 3)  # , sharey='all')
            for j in range(3):
                # if j == 1:
                #     ax[j].imshow(np.log(data[j]))
                # else:
                ax[j].matshow(data[j], extent=(xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()))
                ax[j].title.set_text(names[j])
            # fig.savefig(f'example {_ + 1}.jpg')
            plt.show()
        # data = [imdata, kdata]
        # In a future update this should change to be saved as a numpy file.
        # Numpy files load and save faster, while taking slightly more storage space.
        # filename = f'{save_dir}/lens_set_{str(i + 1)}.xz'
        # with lzma.open(filename, mode='xb') as file:
        #     pickle.dump(data, file)
        # In a future update a validation file can be separated from the training set at this stage


if __name__ == '__main__':
    main()
