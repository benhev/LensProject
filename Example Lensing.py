# import standard python libraries
import numpy as np
import scipy
import os
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util

'''
# define the lens model of the main deflector
main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
kwargs_lens_main = {'theta_E': 1., 'e1': 0.1, 'e2': 0, 'center_x': 0, 'center_y': 0}
kwargs_shear = {'gamma1': 0.05, 'gamma2': 0}
lens_model_list = [main_halo_type, 'SHEAR']
kwargs_lens_list = [kwargs_lens_main, kwargs_shear]

subhalo_type = 'TNFW'  # We chose spherical NFW profiles, feel free to chose whatever you want.

# as an example, we render some sub-halos with a very simple distribution to be added on the main lens
num_subhalo = 10  # number of subhalos to be rendered
# the parameterization of the NFW profiles are:
# - Rs (radius of the scale parameter Rs in units of angles)
# - theta_Rs (radial deflection angle at Rs)
# - center_x, center_y, (position of the centre of the profile in angular units)

Rs_mean = 0.1
Rs_sigma = 0.1  # dex scatter
theta_Rs_mean = 0.05
theta_Rs_sigma = 0.1 # dex scatter
r_min, r_max = -2, 2

Rs_list = 10**(np.log10(Rs_mean) + np.random.normal(loc=0, scale=Rs_sigma, size=num_subhalo))
theta_Rs_list = 10**(np.log10(theta_Rs_mean) + np.random.normal(loc=0, scale=theta_Rs_sigma, size=num_subhalo))
center_x_list = np.random.uniform(low=r_min, high=r_max,size=num_subhalo)
center_y_list = np.random.uniform(low=r_min, high=r_max,size=num_subhalo)
for i in range(num_subhalo):
    lens_model_list.append(subhalo_type)
    kwargs_lens_list.append({'alpha_Rs': theta_Rs_list[i], 'Rs': Rs_list[i],
                             'center_x': center_x_list[i], 'center_y': center_y_list[i],
                            'r_trunc': 5*Rs_list[i]
                            })
'''
# now we define a LensModel class of all the lens models combined
from lenstronomy.LensModel.lens_model import LensModel

kwargs_lens = [{'theta_E': 1, 'e1': 0, 'e2': 0.5, 'center_x': 0, 'center_y': 0}]
lens_model_list = ['SIE']
lensModel = LensModel(lens_model_list)
# we set up a grid in coordinates and evaluate basic lensing quantities on it
x_grid, y_grid = util.make_grid(numPix=100, deltapix=0.05)
kappa = lensModel.kappa(x_grid, y_grid, kwargs_lens)
# we make a 2d array out of the 1d grid points
kappa = util.array2image(kappa)
# and plot the convergence of the lens model

#plt.matshow(np.log10(kappa), origin='lower')
#plt.show()

# we define a very high resolution grid for the ray-tracing (needs to be checked to be accurate enough!)
numPix = 100  # number of pixels (low res of data)
deltaPix = 0.05  # pixel size (low res of data)
high_res_factor = 3  # higher resolution factor (per axis)
# make the high resolution grid
theta_x_high_res, theta_y_high_res = util.make_grid(numPix=numPix*high_res_factor, deltapix=deltaPix/high_res_factor)
# ray-shoot the image plane coordinates (angles) to the source plane (angles)
beta_x_high_res, beta_y_high_res = lensModel.ray_shooting(theta_x_high_res, theta_y_high_res, kwargs=kwargs_lens_list)

# now we do the same as in Section 2, we just evaluate the shapelet functions in the new coordinate system of the source plane
# Attention, now the units are not pixels but angles! So we have to define the size and position.
# This is simply by chosing a beta (Gaussian width of the Shapelets) and a new center

source_lensed = shapeletSet.function(beta_x_high_res, beta_y_high_res, coeff_ngc, n_max, beta=.05, center_x=0.2, center_y=0)
# and turn the 1d vector back into a 2d array
source_lensed = util.array2image(source_lensed)  # map 1d data vector in 2d image

f, ax = plt.subplots(1, 1, figsize=(16, 4), sharex=False, sharey=False)
im = ax.matshow(source_lensed, origin='lower')
ax.set_title("lensed source")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)
plt.show()