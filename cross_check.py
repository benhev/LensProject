import matplotlib.pyplot as plt
import numpy as np
import glob
from os.path import basename
from pathlib import Path


# from lensing_simulation import grid


def theta(npix, deltapix, alpha_x, alpha_y, brightness):
    # X, Y = grid(npix=npix, deltapix=deltapix, origin_ra=0, origin_dec=0).pixel_coordinates
    # print(X.shape, Y.shape)
    alpha_x, alpha_y = np.array([alpha_x, alpha_y]) / deltapix
    image = np.zeros((npix, npix))
    for i in range(npix):
        for j in range(npix):
            beta_x_temp = int(np.around(i - alpha_x[j][i]))
            beta_y_temp = int(np.around(j - alpha_y[j][i]))

            if 0 <= beta_x_temp < npix and 0 <= beta_y_temp < npix:
                if brightness[beta_y_temp][beta_x_temp] > np.mean(brightness):
                    # if np.linalg.norm(np.array([beta_x_temp - 75, beta_y_temp - 75])) < 2:
                    image[j][i] = brightness[beta_y_temp][beta_x_temp]

    return image


def my_get(seek: list, dic: dict, default=None):
    for key in dic.keys():
        key = str(key)
        for phrase in seek:
            if phrase.lower() in key.lower():
                return dic.get(key)
    return default


def my_reshape(arr):
    if len(arr.shape) == 3 and arr.shape[2] == 1:
        return arr.reshape(arr.shape[:-1])
    else:
        return arr


def get_data(data: dict):
    alpha_x = my_reshape(my_get(seek=['alpha_x'], dic=data))
    alpha_y = my_reshape(my_get(seek=['alpha_y'], dic=data))
    kappa = my_reshape(my_get(seek=['convergence', 'kappa'], dic=data))
    brightness = my_reshape(my_get(seek=['galaxy', 'brightness'], dic=data))
    image = my_reshape(my_get(seek=['image'], dic=data))
    dataset = [image, kappa, brightness, alpha_x, alpha_y]
    # if all(dataset):
    return dataset
    # else:
    #     raise KeyError('One or more keys not found in files.')


def main():

    flagged = ['Lens' + str(i) for i in [7, 13, 16, 18, 19]]
    for folder in glob.glob('debug/cross check/debug data/Lens*/'):
        files = glob.glob(folder + '*.npy')
        data = {}
        for file in files:
            data.update({basename(file).removesuffix('.npy').lower().replace(' ', '_').replace('.', ''): np.load(file)})
        image, true_kappa, brightness, alpha_x, alpha_y = get_data(data=data)
        fig, (ax1, ax2) = plt.subplots(2, 3)
        ax1[0].imshow(image, origin='lower')
        ax1[0].title.set_text('Lenstronomy image')
        ax2[0].imshow(theta(npix=150, deltapix=0.1, alpha_x=alpha_x, alpha_y=alpha_y, brightness=brightness),
                      origin='lower')
        ax2[0].title.set_text('Our image')
        ax1[2].imshow(brightness, origin='lower')
        ax1[2].title.set_text('surface brightness')
        ax1[1].imshow(true_kappa, origin='lower')
        ax1[1].title.set_text('Lenstronomy convergence')
        kappa = Kappa(alpha_x, alpha_y, deltapix=0.1)
        ax2[1].imshow(kappa, origin='lower')
        ax2[1].title.set_text('Our convergence')
        ax2[2].imshow(np.abs(kappa - true_kappa),origin='lower')
        ax2[2].title.set_text('Convergence abs difference')

        fig.set_size_inches(12, 8)
        fig.suptitle(Path(folder).name + (' flagged' if Path(folder).name in flagged else ''))
        fig.savefig(f'debug/cross check/{Path(folder).name}.jpg', bbox_inches='tight')
        plt.close(fig)


def Kappa(alpha_x, alpha_y, deltapix):
    f_xx, f_yy = np.zeros_like(alpha_x), np.zeros_like(alpha_y)
    for i in range(alpha_x.shape[1] - 1):
        f_xx[:, i] = (alpha_x[:, i + 1] - alpha_x[:, i]) / deltapix
        f_yy[i, :] = (alpha_y[i + 1, :] - alpha_y[i, :]) / deltapix
    return 1 / 2 * (f_xx + f_yy)


if __name__ == '__main__':
    main()
