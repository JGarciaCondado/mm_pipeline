import tensorflow as tf
import numpy as np
import os
import sys
from shapely.geometry import Polygon, LineString, Point
sys.path.append('../')
from segmentation import segment_cell, boundary_from_pixelated_mask, smooth_boundary
from calibration.extraction import get_centerline, clean_centerline, debranch_centerline
import matplotlib.pyplot as plt
import matplotlib
np.set_printoptions(threshold=sys.maxsize)
from scipy import signal
from skimage.restoration import wiener

def gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# Parameters
m = 60
p = 6.6
pm = 0.11

# Load cell
directory = 'data/extracted_cells/'
cell = np.load(directory+sorted(os.listdir(directory))[-30])

# Load model
model = tf.keras.models.load_model('segmnet')

# Obtain segmentation
pixelated_mask = segment_cell(cell[0], model)

# Second fluorescence channel
cell_fl = cell[1].astype(np.float32)

# Show two channels
plt.subplot(1,2,1)
plt.imshow(cell[0])
plt.subplot(1,2,2)
plt.imshow(cell[1])
plt.show()

# Normalize with values within bacteria
bacteria_px = np.argwhere(pixelated_mask==1.0)
bac_x, bac_y = zip(*bacteria_px)
cell_fl = (cell_fl-np.min(cell_fl[bac_x, bac_y]))/(np.nanmax(cell_fl[bac_x, bac_y])-np.nanmin(cell_fl[bac_x, bac_y]))

# Filtered cell with weiner filter
cell_filtered = wiener(cell_fl, gkern(30, 4), 20)

# Display fluorescence channel
plt.subplot(1,2,1)
plt.imshow(cell_fl)
plt.subplot(1,2,2)
plt.imshow(cell_filtered)
plt.show()

# Eliminate pixels outside bacteria
bg_x, bg_y = zip(*np.argwhere(pixelated_mask==0.0))
cell_fl[bg_x, bg_y] = np.nan

# Set nans to white
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='white')

# Display image
plt.imshow(cell_fl)

# Obtain pixelated boundary
boundary = boundary_from_pixelated_mask(pixelated_mask)

# Smooth boundary
smoothed_boundary = smooth_boundary(boundary, 5)
plt.plot(smoothed_boundary[:,0], smoothed_boundary[:, 1], color='r')

# Obtain centerline
centerline = get_centerline(Polygon(smoothed_boundary).buffer(0))
centerline = clean_centerline(centerline)
centerline = debranch_centerline(centerline)
spline = np.sort(np.array(centerline[0], dtype=([('xcoor', float), ('ycoor', float)])),order='ycoor')
spline = [point for point in spline]
spline_x, spline_y = zip(*spline)
spline = LineString(spline)
plt.plot(spline_x, spline_y, color='k')
plt.show()

# Obtain estimate of r
r = 0.0
for point in boundary:
    r += pm*spline.distance(Point(point))/boundary.shape[0]

# Volume of deepest section
v_max = 2*r*pm**2

# Obtain volume elements
volume_elements = np.empty(cell_fl.shape)
volume_elements[:] = np.nan
for px in bacteria_px:
    pixel = Point((px[1], px[0]))
    # Calculate distance to pixel from centerline
    d2c = pm*spline.distance(pixel)
    # If bigger than radius ignore that pixel
    if d2c < r:
        v = 2*np.sqrt(r**2-d2c**2)*pm**2
        # Avoid very small volumes as well
        if v > v_max*0.2:
            volume_elements[px[0], px[1]] = 1/v

# Plot channels of interest
plt.subplot(1,3,1)
plt.imshow(cell_fl)
plt.subplot(1,3,2)
cell_fl = cell_fl*volume_elements
plt.imshow(cell_fl, vmax=100)
plt.subplot(1,3,3)
cell_filtered = cell_filtered*volume_elements
plt.imshow(cell_filtered, vmax=4)
plt.show()


# Sort pixels by values
px_val = [value for value in cell_filtered.flatten() if not np.isnan(value)]
plt.hist(px_val)
plt.show()

# R values of top quantile
px_val_sort = np.sort(px_val)
top_quantile = px_val_sort[-int(0.25*len(px_val_sort)):]
plt.hist(top_quantile)
plt.show()

distance = []
for index in np.argwhere(cell_filtered>=top_quantile[0]):
    coor = Point((index[1], index[0]))
    # Calculate distance to pixel from centerline
    d2c = pm*spline.distance(coor)/r
    distance.append(d2c)

print(np.mean(distance))
plt.hist(distance)
plt.show()


# R values of botoom quantile
px_val_sort = np.sort(px_val)
bttm_quantile = px_val_sort[:int(0.25*len(px_val_sort))]
plt.hist(bttm_quantile)
plt.show()

distance = []
for index in np.argwhere(cell_filtered<=bttm_quantile[-1]):
    coor = Point((index[1], index[0]))
    # Calculate distance to pixel from centerline
    d2c = pm*spline.distance(coor)/r
    distance.append(d2c)

print(np.mean(distance))
plt.hist(distance)
plt.show()
