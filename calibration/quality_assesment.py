import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from models import Microscope, SpherocylindricalBacteria
from extraction import extract_all_params
from comparison import MSE, l2_norm, MSSIM


ex_wv = 0.8  # emmitted wavelength by microscope for excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
pixel_size = 4.4  # pixel size
NA = 0.95  # Numerical aperture
magnification = 40  # magnification
microscope = Microscope(magnification, NA, ex_wv, em_wv, pixel_size)

trench_reconstruction = []
bacterias = []
centroids = []
prev_shape = 0.0
for n in range(4):
    im = np.load('Cell_test_%s.npy' % n)
    rm_indices = np.where(im==0.0)[0]
    im = np.delete(im, rm_indices, axis=0)
    l, R, theta, centroid = extract_all_params(im, 40, 4.4)
    r = 0.5
    bacteria = SpherocylindricalBacteria(l-r, r, R, theta, 1700,  ex_wv, em_wv)
    bacterias.append(bacteria)
    if n!=0:
        centroid = [centroid[0], centroid[1]]
        centroid[1] = centroid[1] + prev_shape
    centroids.append(centroid)
    trench_reconstruction.append(im)
    prev_shape+=im.shape[0]

trench_reconstruction = np.concatenate(trench_reconstruction)
image = microscope.image_trench(bacterias, centroids, trench_reconstruction.shape, sigma=3.1)
trench_reconstruction = (trench_reconstruction-np.min(trench_reconstruction))/(np.max(trench_reconstruction)-np.min(trench_reconstruction))
image = (image-np.min(image))/(np.max(image)-np.min(image))
print(MSE(trench_reconstruction, image))
print(l2_norm(trench_reconstruction, image))
print(MSSIM(trench_reconstruction, image))
plt.subplot(1,2,1)
plt.imshow(trench_reconstruction)
plt.subplot(1,2,2)
plt.imshow(image)
plt.show()
