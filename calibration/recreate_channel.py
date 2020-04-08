import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from extraction import extract_cell_info
from comparison import compare_images
from bacteria_model import SpherocylindricalBacteria
from microscope_models import Microscope
from scipy.ndimage.filters import gaussian_filter

ex_wv = 0.8  # emmitted wavelength by microscope for excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
pixel_size = 4.4  # pixel size
NA = 0.95  # Numerical aperture
magnification = 40  # magnification
microscope = Microscope(magnification, NA, ex_wv, em_wv, pixel_size)

trench_reconstruction = []
cell_info = []
r_s = []
for n in range(4):
    im = np.load('Cell_test_%s.npy' % n)
    rm_indices = np.where(im==0.0)[0]
    im = np.delete(im, rm_indices, axis=0)
    trench_reconstruction.append(im)
    info = extract_cell_info('Cell_test_%s.npy' % n)
    cell_info.append(info)
    if info[-1] != None:
        r_s.append(info[-1])
#    plt.imshow(im)
#    plt.show()

# TODO take into acoount when reconstruction trench original shape of trench

average_r = np.average(r_s)

total_length = 0
bacterias = []
centroids = []
for length, centroid, theta, l, R, r in cell_info:
    if r == None:
        r = average_r
    l = l-r
    centroid = (centroid[0], centroid[1] + total_length)
    centroids.append(centroid)
    bacterias.append(SpherocylindricalBacteria(r, l, R, theta, ex_wv, em_wv, 1000))
    total_length += length

trench = np.load('Cropped_test_channel.npy')
print(trench.shape)

fake_trench = microscope.image_trench(bacterias, centroids, total_length)

trench_reconstruction = np.concatenate(trench_reconstruction)
print(trench_reconstruction.shape)

print(compare_images(trench_reconstruction, fake_trench))

fig, axs = plt.subplots(1, 3, constrained_layout=True)
axs[0].imshow(trench)
axs[0].set_title('original')
axs[1].imshow(trench_reconstruction)
axs[1].set_title('decomposed')
axs[2].imshow(fake_trench)
axs[2].set_title('fake')
plt.show()

trench = gaussian_filter(trench, sigma=1)
trench_reconstruction = gaussian_filter(trench_reconstruction, sigma=1)
fake_trench = gaussian_filter(fake_trench, sigma=1)

print(compare_images(trench_reconstruction, fake_trench))

fig, axs = plt.subplots(1, 3, constrained_layout=True)
axs[0].imshow(trench)
axs[0].set_title('original')
axs[1].imshow(trench_reconstruction)
axs[1].set_title('decomposed')
axs[2].imshow(fake_trench)
axs[2].set_title('fake')
plt.show()
