from models import SpherocylindricalBacteria, Microscope
import numpy as np
import matplotlib.pyplot as plt

# Ball park values
ex_wv = 0.8  # emmitted wavelength by microscope for excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
pixel_size = 4.4  # pixel size
NA = 0.95  # Numerical aperture
m = 40  # magnificatio

l = 4.0
r = 0.45
R = 1000
theta = 0
density = 5000

shape  = (50, 26)

bacteria = SpherocylindricalBacteria(l, r, R, theta, density, ex_wv, em_wv)
#bacteria.plot_3D()
bacteria.plot_2D()

microscope = Microscope(m, NA, ex_wv, em_wv, pixel_size)
im = microscope.image_bacteria(bacteria, (15,25), shape, 0.01, noise=0)
im = microscope.image_bacteria_ground_truth(bacteria, (15,25), shape)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        if im[i,j] == 1.0:
            im[i,j] = np.random.normal(8)
microscope.display_image(im)
microscope.display_image_with_boundary(im, bacteria, (15, 25))

bacterias = []
centroids = []
centroid_height = 0.0
half_length = 0.0
for i in range(4):
    centroid_height += half_length
    spacing = np.random.uniform(0, 5)
    centroid_height += spacing
    l = np.random.uniform(2, 4)
    r = np.random.uniform(0.45, 0.55)
    R = np.random.uniform(5, 100)
    theta = np.random.uniform(-15, 15)
    bacterias.append(SpherocylindricalBacteria(l, r, R, theta, density, ex_wv, em_wv))
    half_length = m*(l/2+r)/pixel_size
    centroid_height += half_length
    centroids.append((13, centroid_height))

shape = (200, 26)
im = microscope.image_trench(bacterias, centroids, shape, 4.42, noise = 150)
microscope.display_image(im)

n_trenches = 16
bacteria_array = []
centroid_array = []
for i in range(n_trenches):
    bacterias = []
    centroids = []
    centroid_height = 0.0
    half_length = 0.0
    n_bacteria = int(np.random.normal(4.2, 0.2))
    for i in range(n_bacteria):
        centroid_height += half_length
        spacing = np.random.normal(3)
        centroid_height += spacing
        l = np.random.uniform(4, 7)
        r = np.random.uniform(0.45, 0.55)
        R = np.random.uniform(5, 100)
        theta = np.random.uniform(-15, 15)
        bacterias.append(SpherocylindricalBacteria(l, r, R, theta, density, ex_wv, em_wv))
        half_length = m*(l/2+r)/pixel_size
        centroid_height += half_length
        centroids.append((0, centroid_height))
    bacteria_array.append(bacterias)
    centroid_array.append(centroids)
shape = (300, 700)
im1 = microscope.image_MM(bacteria_array, centroid_array, shape, 5.0, 4.42, noise = 200)
microscope.display_image(im1)
#im2 = microscope.image_MM_sampling(bacteria_array, centroid_array, shape, 4.0, 4.42, noise = 150)
#plt.subplot(1, 2, 1)
#plt.imshow(im1)
#plt.subplot(1, 2, 2)
#plt.imshow(im2)
#plt.show()
