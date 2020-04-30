import numpy as np
import matplotlib.pyplot as plt
import sys
from extraction import horizontal_probability_density
sys.path.append('../')

from models import SpherocylindricalBacteria, Microscope

# Ball park values
ex_wv = 0.8  # emmitted wavelength by microscope for excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
pixel_size = 4.4  # pixel size
NA = 0.95  # Numerical aperture
m = 40  # magnificatio

l = 5
r = 0.6
R = 1000
theta = 0
density = 1700

shape  = (40, 26)

bacteria = SpherocylindricalBacteria(l, r, R, theta, density, ex_wv, em_wv)
microscope = Microscope(m, NA, ex_wv, em_wv, pixel_size)
im = microscope.image_bacteria_sampling(bacteria, (13,20), shape, 4.42, noise=150)
plt.imshow(im)
plt.show()
counts = horizontal_probability_density(im)
bins = np.arange(0, len(counts)+1)-0.5
std_dev = 4.42
g_x = np.exp(-(np.arange(0, len(counts))-13)**2/(2*std_dev**2))/(std_dev*np.sqrt(2*np.pi))
g_x = g_x/np.sum(g_x)
plt.hist(bins[:-1], bins, weights=counts, label="Experimental" )
plt.hist(bins[:-1], bins, weights=g_x, label="Gaussian fit", alpha=0.75)
plt.show()
