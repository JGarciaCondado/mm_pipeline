import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from extraction import get_centerline, clean_centerline, debranch_centerline, calculate_theta, calculate_l, convert_centerline, calculate_R, calculate_r, horizontal_probability_density, fwhm, grid_search_r_sigma, extract_density_photons
from rmatching import match_r_and_psf, convolved_circle_px, cauchy, convolve_cauchy, optimum_r_and_psf, optimum_cauchy, model_cauchy, model

#TODO use new BacteriaModel

#TODO check how many of this do we need
from contour import contour_real
from shapely.geometry import Polygon
from centerline.geometry import Centerline
from bacteria_model import SpherocylindricalBacteria
from microscope_models import Microscope
from scipy import ndimage, misc
from tqdm import tqdm
from scipy.interpolate import splrep, sproot, splev
from scipy.optimize import curve_fit
from microscope_models import Microscope
from models import Microscope, SpherocylindricalBacteria

#TODO label an clean up all of the graphs + titles
#TODO also nicely print all values it calculates


#TODO create a single array/tuple to hold all parameters so can be fed easily to other functions

# Microscope constants
pixel_size = 4.4  # pixel size
NA = 0.95  # Numerical aperture
m = 40  # magnification
ex_wv = 0.8
em_wv = 0.59

#Load image
cell = np.load("Cell_test_1.npy")
rm_indices = np.where(cell==0.0)[0]
cell = np.delete(cell, rm_indices, axis=0)

#shape of image
shape = cell.shape
length = shape[0]

#Obtain contour of cell
contour = contour_real(cell, 1.0)
active_contour = contour.active_contour
boundary = Polygon(active_contour).buffer(0) #Fixes any holes in polygon
[minx, miny, maxx, maxy] = boundary.bounds

#Centroid of cell
centroid = (boundary.centroid.x, boundary.centroid.y)

#Obtain centerline
centerline = get_centerline(boundary)

#Plot image with centerline and boundary
fig, ax = plt.subplots()
ax.plot(contour.active_contour[:, 0], contour.active_contour[:, 1], 'y', lw=1, label="Active Contour")

#Plot line segments that make up the centerline 
for line in list(centerline.geoms):
    x, y = zip(*list(line.coords))
    ax.plot(x, y, 'r')
ax.plot(x, y, 'r', label = 'centerline')

#TODO centerline split into its components with names
#Clean centerline
centerline = clean_centerline(centerline)

#Place end_points and branches
coor_x, coor_y = zip(*centerline[0])
ax.scatter(coor_x, coor_y, c='b', label="end points")

coor_x, coor_y = zip(*centerline[2])
ax.scatter(coor_x, coor_y, c='g', label="branch points")

#Debranch centerline
centerline = debranch_centerline(centerline)

#Plot removed coordinates
spline_x, spline_y = zip(*centerline[2])
ax.scatter(spline_x, spline_y, c='m', label="debranch points")

#Plot over original image
plt.imshow(cell, origin='lower')
plt.title("Cell boundary and centerline")
plt.legend()
plt.show()

#TODO make sure that image shown has (0,0) in botom left hand corner for all

#Plot spline left
[spline, [max_coor, min_coor], remove_coords] = centerline
y_max, y_min = int(max_coor[1]), int(min_coor[1])
x, y = zip(*spline)
plt.scatter(x, y)
plt.title("Spline points")
plt.ylabel('Pixels')
plt.xlabel('Pixels')
plt.axis('scaled')
plt.show()

#Obtain theta
theta = calculate_theta(centerline)

#Obtain l
l = calculate_l(centerline, pixel_size, m)

#Reduce centerline to new coordinate system
centerline = convert_centerline(centerline, theta, pixel_size, m, l)

#Calculate R
R = calculate_R(centerline, l)

#Show spline
x, y = zip(*centerline)
plt.scatter(x, y)
x = np.arange(-l/2, l/2, 0.01)
def fn(R, x):
    return np.sqrt(R**2 - x**2) - np.sqrt(R**2-(l**2)/4)
plt.plot(x, [fn(R, i) for i in x], 'k')
plt.title('Curvature fit')
plt.ylabel('micrometers')
plt.xlabel('micrometers')
plt.axis('scaled')
plt.show()

#TODO example that has big R seperately
#Plot rotated image
cell_rot = ndimage.rotate(cell, -theta, reshape=False)
cell_rot = cell_rot.astype('float')
cell_rot[cell_rot == 0.0] = np.nan
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cell)
plt.subplot(1, 2, 2)
plt.title("Rotated Image")
plt.imshow(cell_rot)
plt.show()

#Plot image with caps taken off
#TODO y_min and y_max have they been rotated properly
#TODO l is too big
plt.title("Rotated and cropped caps")
plt.imshow(cell_rot[y_min:y_max, :])
plt.show()

#Extract density
print("Photon density: {}".format(extract_density_photons(cell_rot[y_min:y_max, :], 4.4, 40, 0.5, 200)))


#Plot probability density histogram
original_px = horizontal_probability_density(cell_rot[y_min:y_max, :])
bins = np.arange(-0.5, len(original_px))
plt.hist(bins[:-1], bins, weights=original_px, label="Discrete Experimental PDF")

#Spline approximation of histogram
spl = splrep(range(len(original_px)), original_px)
xorg = np.linspace(0, len(original_px), 100)
yorg = splev(xorg, spl)
h = xorg[np.argmax(yorg)]
plt.plot(xorg, yorg, label="B-spline approximation of PDF")
plt.legend()
plt.title("Cell horizontal experimental PDF with noise removed")
plt.show()

# Calculate r without matching PSF
r = calculate_r(cell, l, R, theta, shape, y_max, y_min, centroid)

# Image bacteria
bacteria = SpherocylindricalBacteria(l-r,r, R, theta, 800,  ex_wv, em_wv)
microscope = Microscope(m, NA, ex_wv, em_wv, pixel_size)
image = microscope.image_bacteria(bacteria, centroid, shape)
im_rot = ndimage.rotate(image, -theta, reshape=False)
im_rot = im_rot.astype('float')
im_rot[im_rot == 0.0] = np.nan


#Compare distribution
image_model_px = horizontal_probability_density(im_rot[y_min:y_max, :])
plt.hist(bins[:-1], bins, weights=image_model_px, label="Model Discrete Experimental PDF")
spl = splrep(range(len(image_model_px)), image_model_px)
xim = np.linspace(0, len(image_model_px), 100)
yim = splev(xim, spl)
plt.plot(xim, yim, label="Model B-spline approximation of PDF")
plt.plot(xorg, yorg, label="Experimental B-spline approximation of PDF")
plt.title('Model image horizontal PDF with noise removed and matched FWHM')
#TODO fix axe positioning
plt.legend()
plt.show()


#Images
plt.subplot(1,2,1)
plt.title("Original image")
plt.imshow(cell)
plt.subplot(1,2,2)
plt.title("Model image")
plt.imshow(image)
plt.show()

#Get optmal gaussian
x = range(len(original_px))
def gaus(x, x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

mu_org = np.sum([px*i for i, px in enumerate(original_px)])
std_dev_org = np.sum([px*(i-mu_org)**2 for i, px in enumerate(original_px)])
org_fwhm = fwhm(x, original_px)
popt, pcov = curve_fit(gaus, x, original_px, p0 = [h, std_dev_org])
poptc, pcovc = curve_fit(cauchy, x, original_px, p0 = [org_fwhm/2, h])

# Plot all different distribution
plt.plot(xorg, yorg, label='original')
plt.plot(np.arange(0, 26, 0.01), gaus(np.arange(0,26,0.01), popt[0], popt[1]), label='gaussian best fit')
plt.plot(np.arange(0, 26, 0.01), cauchy(np.arange(0,26,0.01), org_fwhm/2, poptc[1]), label='cauchy matched fwhm')
plt.plot(np.arange(0, 26, 0.01), cauchy(np.arange(0,26,0.01), poptc[0], poptc[1]), label='cauchy best fit')
plt.legend()
plt.show()

# Plot all different distribution
rc, gc, hc = optimum_cauchy(original_px)
rg, gg, hg = optimum_r_and_psf(original_px)
plt.plot(xorg, yorg, label='original')
plt.plot(np.arange(0, 26, 0.01), model_cauchy(np.arange(0,26, 0.01), rc, hc, gc), label="cauchy model")
plt.plot(np.arange(0, 26, 0.01), model(np.arange(0,26, 0.01), rg, hg, gg), label="gaussian model")
plt.legend()
plt.show()
#Find optimum with density matching
opt_r, opt_sigma = grid_search_r_sigma(original_px, l, R, theta, shape, y_max, y_min, centroid, microscope, ex_wv, em_wv)
# Image bacteria
bacteria = SpherocylindricalBacteria(l-opt_r,opt_r, R, theta, 1700,  ex_wv, em_wv)
microscope = Microscope(m, NA, ex_wv, em_wv, pixel_size)
image = microscope.image_bacteria(bacteria, centroid, shape, sigma=opt_sigma)
im_rot = ndimage.rotate(image, -theta, reshape=False)
im_rot = im_rot.astype('float')
im_rot[im_rot == 0.0] = np.nan


#Compare distribution
image_model_px = horizontal_probability_density(im_rot[y_min:y_max, :])
plt.hist(bins[:-1], bins, weights=image_model_px, label="Model Discrete Experimental PDF")
spl = splrep(range(len(image_model_px)), image_model_px)
xim = np.linspace(0, len(image_model_px), 100)
yim = splev(xim, spl)
plt.plot(xim, yim, label="Model B-spline approximation of PDF")
plt.plot(xorg, yorg, label="Experimental B-spline approximation of PDF")
plt.title('Model image horizontal PDF with noise removed and optimum r and sigma')
#TODO fix axe positioning
plt.legend()
plt.show()


bacteria = SpherocylindricalBacteria(l-rc/11,rc/11, R, theta, 2000,  ex_wv, em_wv)
im_c = microscope.image_bacteria_cauchy(bacteria, centroid, shape, gamma=gc, noise=200)

#Images
plt.subplot(2,3,1)
plt.title("Original image")
plt.imshow(cell)
plt.subplot(2,3,2)
plt.title("Model image, gaussian")
plt.imshow(image)
plt.subplot(2,3,3)
plt.title("Model image, cauchy")
plt.imshow(im_c)
cell = (cell-np.min(cell))/(np.max(cell)-np.min(cell))
image = (image-np.min(image))/(np.max(image)-np.min(image))
im_c = (im_c-np.min(im_c))/(np.max(im_c)-np.min(im_c))
plt.subplot(2,3,4)
plt.imshow(cell)
plt.subplot(2,3,5)
plt.imshow(image)
plt.subplot(2,3,6)
plt.imshow(im_c)
plt.show()


#Plot image with measured boundary and new boundary
fig, ax = plt.subplots()
ax.plot(contour.active_contour[:, 0], contour.active_contour[:, 1], 'y', lw=1, label="Active Contour")
bacteria_boundary = np.array(list(map(list, bacteria.boundary)))
verts = microscope._transform_vertices(bacteria_boundary, bacteria, centroid)
ax.plot(verts[:,0], verts[:, 1], 'g', lw=1, label="Ground Truth")
plt.imshow(cell)
plt.show()
