
from scipy import ndimage as ndi

from skimage import feature

import numpy as np
from skimage.io import imread, imsave
from time import time
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import pylab as pl
from skimage.filters import gaussian
from skimage.transform import rescale
import random
from spline_registration.transform_models import ElasticTransform
from spline_registration.utils import coordenades_originals, create_results_dir,rescalar_imatge, color_a_grisos
from spline_registration.losses import RMSE, info_mutua
from scipy.optimize import least_squares, minimize


imatge_input = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

resolucio_ideal = (250,250)
scale_factor = (resolucio_ideal[0] / imatge_input.shape[0], resolucio_ideal[1] / imatge_input.shape[1])
imatge_input = rescale(imatge_input, scale_factor, multichannel=True, anti_aliasing=False)
imatge_reference = rescale(imatge_reference, scale_factor, multichannel=True, anti_aliasing=False)

imatge_input_gris = color_a_grisos(imatge_input)
imatge_reference_gris = color_a_grisos(imatge_reference)

sigma = 5

imatge_input_gaussian_gris = gaussian(imatge_input_gris, sigma= sigma, multichannel=False)

im1 = imatge_input_gris
im2 = imatge_reference_gris

edges1 = feature.canny(im1, sigma = sigma)
edges2 = feature.canny(im2, sigma = sigma)






















fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(imatge_reference_gris , cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title(r'Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title(r'Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()

