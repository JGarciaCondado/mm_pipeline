import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from contour import contour_real
from shapely.geometry import Polygon
from centerline.geometry import Centerline
from bacteria_model import SpherocylindricalBacteria
from microscope_models import Microscope
from scipy import ndimage, misc
from tqdm import tqdm

cell = np.load("Cell_test_2.npy")
rm_indices = np.where(cell==0.0)[0]
cell = np.delete(cell, rm_indices, axis=0)
shape = cell.shape
contour = contour_real(cell, 1.0)
active_contour = contour.active_contour
boundary = Polygon(active_contour).buffer(0)
centroid = (boundary.centroid.x, boundary.centroid.y)
[minx, miny, maxx, maxy] = boundary.bounds
print(centroid)
centerline = Centerline(boundary)
fig, ax = plt.subplots()
ax.plot(contour.active_contour[:, 0], contour.active_contour[:, 1], 'y', lw=1, label="Active Contour")
end_points = []
spline = []
branch = []
for line in list(centerline.geoms):
    coordinates = list(line.coords)
    for coor in coordinates:
        if coor not in branch:
            if coor not in spline:
                if coor not in end_points:
                    end_points.append(coor)
                else:
                    end_points.remove(coor)
                    spline.append(coor)
            else:
                spline.remove(coor)
                branch.append(coor)
    x, y = zip(*coordinates)
    ax.plot(x, y, 'r')

for coor in end_points:
    ax.scatter(coor[0], coor[1], c='b')

for coor in branch:
    ax.scatter(coor[0], coor[1], c='g')


min_coor = (1000.0, 1000.0)
max_coor = (0.0, 0.0)
for branch_coor in branch:
    if branch_coor[1] < min_coor[1]:
        min_coor = branch_coor
    if branch_coor[1] > max_coor[1]:
        max_coor = branch_coor
end_branches = [max_coor, min_coor]
print(min_coor)

if branch != []:
    for end_coor in end_points:
        distanceToBranch = 1000
        closest_branch = 0.0
        for branch_coor in branch:
            if distanceToBranch > np.sqrt((end_coor[0]-branch_coor[0])**2 + (end_coor[1]-branch_coor[1])**2):
                distanceToBranch = np.sqrt((end_coor[0]-branch_coor[0])**2 + (end_coor[1]-branch_coor[1])**2)
                close_branch = branch_coor
        if distanceToBranch < 4.0 or close_branch not in end_branches:
            remove_coor = []
            for spline_coor in spline:
                if np.sqrt((end_coor[0]-spline_coor[0])**2 + (end_coor[1]-spline_coor[1])**2) < distanceToBranch:
                    remove_coor.append(spline_coor)
                    ax.scatter(spline_coor[0], spline_coor[1], c='m')
            for coor in remove_coor:
                spline.remove(coor)
        # need to create then endpoints as branch // maybe pop the end_points?
        else:
            if end_coor[1] < min_coor[1]:
                min_coor = end_coor
            if end_coor[1] > max_coor[1]:
                max_coor = end_coor
else:
    branch = end_points

print(min_coor)

plt.imshow(cell)
plt.show()

for branch_coor in branch:
    spline.append(branch_coor)

m = 40
pixel_size = 4.4

x, y = zip(*spline)
plt.scatter(x, y)
plt.axis('scaled')
plt.show()

theta = np.arctan((max_coor[0]-min_coor[0])/(max_coor[1]-min_coor[1]))
print(theta*180/np.pi)
rotation_matrix = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
l = (max_coor[1]-min_coor[1])*pixel_size/m
print(l)
for i, coor in enumerate(spline):
    new_coor = np.subtract(coor,min_coor)
    new_coor = rotation_matrix.dot(new_coor)
    new_coor = new_coor*pixel_size/m
    new_coor[1] = new_coor[1]-l/2
    new_coor = new_coor[::-1]
    spline[i] = new_coor

def fn(R, x):
    return np.sqrt(R**2 - x**2) - np.sqrt(R**2-(l**2)/4)

R = 100
n = 100
dr = 5
old_error = 10000000
for i in range(n):
    error = 0.0
    for x,y in spline:
        error += abs(fn(R, x) - y)
    if error > old_error:
        dr = -dr/2
    old_error = error
    R = R-dr
print(R)

#Show spline
x, y = zip(*spline)
plt.scatter(x, y)
x = np.arange(-l/2, l/2, 0.01)
plt.plot(x, [fn(R, i) for i in x], 'k')
plt.axis('scaled')
plt.show()

ex_wv = 0.8  # emmitted wavelength by microscope for excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
pixel_size = 4.4  # pixel size
NA = 0.95  # Numerical aperture
magnification = 40  # magnification
from scipy.interpolate import splrep, sproot, splev

class MultiplePeaks(Exception): pass
class NoPeaksFound(Exception): pass

def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = np.max(y)/2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
    else:
        return abs(roots[1] - roots[0])

r_final = 0.5
microscope = Microscope(magnification, NA, ex_wv, em_wv, pixel_size)

if R > l*4:
    print("Adjusting r")
    fig = plt.figure(figsize=(10, 2))
    ax1, ax2 = fig.subplots(1, 2)
    cell_rot = ndimage.rotate(cell, -theta*180/np.pi, reshape=False)
    cell_rot = cell_rot.astype('float')
    cell_rot[cell_rot == 0.0] = np.nan
    ax1.imshow(cell, cmap='gray')
    ax1.set_axis_off()
    ax2.imshow(cell_rot, cmap='gray')
    ax2.set_axis_off()
    fig.set_tight_layout(True)
    plt.show()
    y_max, y_min = int(max_coor[1]), int(min_coor[1])
    print(y_max, y_min)
    plt.imshow(cell_rot[y_min:y_max, :])
    plt.show()
    counts = np.nanmean(cell_rot[y_min:y_max, :], axis=0)
    counts = (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
    plt.hist(np.arange(0, len(counts)), weights=counts, bins=np.arange(-0.5, len(counts)+1))
    spl = splrep(range(len(counts)), counts)
    x2 = np.linspace(0, len(counts), 100)
    y2 = splev(x2, spl)
    plt.plot(x2, y2)
    plt.show()
    print(counts)
    org = fwhm(range(len(counts)), counts)
    print(org)
    r_values = np.arange(0.15, 0.6, 0.01)
    prev_fwhm = 0.0
    for r in tqdm(r_values):
        bacteria = SpherocylindricalBacteria(r, l, R, theta*180/np.pi, ex_wv, em_wv, 10000)
        #TODO optional paddic and also change centroid so calculation so can do cropping before
        image = microscope.image_bacteria_conv(bacteria, centroid, shape)
        rm_indices = np.where(image==0.0)[0]
        image = np.delete(image, rm_indices, axis=0)
        im_rot = ndimage.rotate(image, -theta*180/np.pi, reshape=False)
        im_rot = im_rot.astype('float')
        im_rot[im_rot == 0.0] = np.nan
        counts = np.nanmean(im_rot[y_min:y_max, :], axis=0)
        counts = (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
        current_fwhm = fwhm(range(len(counts)), counts)
        if current_fwhm > org:
            r_final = r - 0.01*((current_fwhm-org)/(current_fwhm-prev_fwhm))
            #TODO plot both together
            plt.hist(np.arange(0, len(counts)), weights=counts, bins=np.arange(-0.5, len(counts)+1))
            spl = splrep(range(len(counts)), counts)
            x2 = np.linspace(0, len(counts), 100)
            y2 = splev(x2, spl)
            plt.plot(x2, y2)
            plt.show()
            print(r_final)
            break
        prev_fwhm = current_fwhm

fig, ax = plt.subplots()
ax.plot(contour.active_contour[:, 0], contour.active_contour[:, 1], 'y', lw=1, label="Active Contour")

bacteria = SpherocylindricalBacteria(r_final, l, R, theta*180/np.pi, ex_wv, em_wv, 3000)
verts = bacteria.boundary[:, :-1]
#verts = verts - bacteria.min[:-1] # move to get non-zero values
verts = verts*m #magnification
verts = verts / pixel_size #scaling by size of pixels
verts = verts + (centroid[1], centroid[0]) # add padding
verts[:,[0, 1]] = verts[:,[1, 0]] #make horizontal
ax.plot(verts[:,0], verts[:, 1], 'g', lw=1, label="Ground Truth")
plt.imshow(cell)
plt.show()
bacteria.plot_2D()

image = microscope.image_bacteria_conv(bacteria, centroid, shape)
rm_indices = np.where(image==0.0)[0]
image = np.delete(image, rm_indices, axis=0)

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1,2)

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
# use unified colormap max value of 700?
cell = (cell-np.min(cell))/(np.max(cell)-np.min(cell))
image = (image-np.min(image))/(np.max(image)-np.min(image))
axarr[0].imshow(cell)
axarr[1].imshow(image)
plt.show()
