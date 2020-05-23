import tensorflow as tf
import numpy as np
import os
import sys
from shapely.geometry import Polygon, LineString, Point
sys.path.append('../')
from segmentation import segment_cell, boundary_from_pixelated_mask, smooth_boundary
from calibration.extraction import get_centerline, clean_centerline, debranch_centerline2
import matplotlib.pyplot as plt

# Parameters
m = 60
p = 6.6
pm = 0.11

# Load cell
directory = 'data/extracted_cells/'
cell = np.load(directory+os.listdir(directory)[3])

# Load model
model = tf.keras.models.load_model('segmnet')

# Obtain segmentation
pixelated_mask = segment_cell(cell[0], model)

# Filter second channel with those pixels that are within the bacteria
cell_fl = cell[1]
cell_fl = (cell_fl-np.min(cell_fl))/(np.max(cell_fl)-np.min(cell_fl))
cell_fl = cell_fl*pixelated_mask
plt.imshow(cell_fl)

# Obtain pixelated boundary
boundary = boundary_from_pixelated_mask(pixelated_mask)

# Smooth boundary
smoothed_boundary = smooth_boundary(boundary, 5)
plt.plot(smoothed_boundary[:,0], smoothed_boundary[:, 1], color='r')

# Obtain centerline
centerline = get_centerline(Polygon(smoothed_boundary).buffer(0))
centerline = clean_centerline(centerline)
centerline = debranch_centerline2(centerline)
spline = centerline[0]
spline_x, spline_y = zip(*spline)
spline = LineString(spline)
plt.scatter(spline_x, spline_y, color='k')

# Obtain estimate of r
r = 0.0
for point in boundary:
    r += pm*spline.distance(Point(point))/boundary.shape[0]

# Volume of deepest section
v_max = 2*r*pm**2

# Obtain volume elements
volume_elements = np.zeros(cell_fl.shape)
bacteria_px = np.argwhere(pixelated_mask==1.0)
print(bacteria_px.shape)
total_volume = 0.0
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
        total_volume += v
print(total_volume)
plt.show()
plt.subplot(1,2,1)
plt.imshow(cell_fl*volume_elements, vmax=100)
plt.subplot(1,2,2)
plt.imshow(cell_fl)
plt.show()
