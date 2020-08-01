import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from segmentation import segment_cell
from coordinate_system import deblur_ch2, normalize_ch2, get_boundary, boundary2centerline, estimate_r, volume_elements, fluorescence_density, quantiles, average_r_quantile
import matplotlib.pyplot as plt

# Load cell
directory = 'data/extracted_cells/'
cells_late = []
for f in sorted(os.listdir(directory))[-400:]:
    cell = np.load(directory+f).astype(np.float64)
    cells_late.append(cell)

# Constants
pm = 0.11

# Load model
model = tf.keras.models.load_model('segmnet')

r_s_late = []
l_s = []
r_s = []
im_l1 = np.zeros((80, 30))
im_l2 = np.zeros((80, 30))
im_l3 = np.zeros((80, 30))
for cell in cells_late:
    # Obtain segmentation
    pixelated_mask = segment_cell(cell[0], model)

    # Normalize ch2
    cell_fl = normalize_ch2(cell[1], pixelated_mask)

    # Deblur channel 2
#    cell_fl = deblur_ch2(cell_fl, 4)

    # Get boundary
    boundary = get_boundary(pixelated_mask)


    # Get centerline
    centerline = boundary2centerline(boundary)
    cx, cy = centerline.centroid.x, centerline.centroid.y
    minx, miny, maxx, maxy = centerline.bounds
    l = pm*np.sqrt((maxx-minx)**2+(maxy-miny)**2)
    l_s.append(l)

    # Estimate r
    r = estimate_r(boundary, centerline, pm)
    r_s.append(r)

    # Volume elements
    ve = volume_elements(pixelated_mask, centerline, r, pm)
#    x, y = zip(*np.argwhere(pixelated_mask==0))
#    cell_fl[x,y] = np.nan

    # Fluorescence density
    cell_fl = fluorescence_density(cell_fl, ve)
    dx, dy = cx-cell_fl.shape[1]/2, cy-cell_fl.shape[0]/2
    cell_fl = np.roll(cell_fl, -int(dy), axis=0)
    cell_fl = np.roll(cell_fl, -int(dx), axis=1)


    if r > 0.435 and r < 0.465 and l>4.0 and l<4.5:
    # Pad cell
        pad = (80 - cell_fl.shape[0]) / 2
        if pad.is_integer():
            cell = np.pad(cell_fl, ((int(pad), int(
                pad)), (0, 0)), mode='constant', constant_values=0)
        else:
            cell = np.pad(cell_fl, ((
                int(pad - 0.5), int(pad + 0.5)), (0, 0)),
                mode='constant', constant_values=0)
        im_l1 += np.nan_to_num(cell)
    if r > 0.435 and r < 0.465 and l>4.5 and l<5.0:
    # Pad cell
        pad = (80 - cell_fl.shape[0]) / 2
        if pad.is_integer():
            cell = np.pad(cell_fl, ((int(pad), int(
                pad)), (0, 0)), mode='constant', constant_values=0)
        else:
            cell = np.pad(cell_fl, ((
                int(pad - 0.5), int(pad + 0.5)), (0, 0)),
                mode='constant', constant_values=0)
        im_l2 += np.nan_to_num(cell)
    if r > 0.435 and r < 0.465 and l>5.0 and l<5.5:
    # Pad cell
        pad = (80 - cell_fl.shape[0]) / 2
        if pad.is_integer():
            cell = np.pad(cell_fl, ((int(pad), int(
                pad)), (0, 0)), mode='constant', constant_values=0)
        else:
            cell = np.pad(cell_fl, ((
                int(pad - 0.5), int(pad + 0.5)), (0, 0)),
                mode='constant', constant_values=0)
        im_l3 += np.nan_to_num(cell)

    # Quantiles
    qs = quantiles(cell_fl, 4)

    # Average r
    r_avg = average_r_quantile(cell_fl, qs[-1], centerline, r, pm)

    r_s_late.append(r_avg)
plt.hist(l_s)
plt.show()
plt.hist(r_s)
plt.show()
plt.subplot(1,3,1)
plt.imshow(im_l1)
plt.subplot(1,3,2)
plt.imshow(im_l2)
plt.subplot(1,3,3)
plt.imshow(im_l3)
plt.show()
