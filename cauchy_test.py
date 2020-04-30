from models import SpherocylindricalBacteria, Microscope
import numpy as np
import matplotlib.pyplot as plt

# Ball park values
ex_wv = 0.8  # emmitted wavelength by microscope for excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
pixel_size = 4.4  # pixel size
NA = 0.95  # Numerical aperture
m = 40  # magnificatio

l = 5
r = 0.5
R = 8
theta = 10
density = 1700

shape  = (50, 26)

bacteria = SpherocylindricalBacteria(l, r, R, theta, density, ex_wv, em_wv)
microscope = Microscope(m, NA, ex_wv, em_wv, pixel_size)
im_c = microscope.image_bacteria_cauchy(bacteria, (15,25), shape, 2.7, noise=150)
im_g = microscope.image_bacteria_sampling(bacteria, (15,25), shape, 3.1, noise=150)
plt.subplot(1,2,1)
plt.imshow(im_c)
plt.subplot(1,2,2)
plt.imshow(im_g)
plt.show()
