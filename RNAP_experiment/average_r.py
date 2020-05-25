import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from segmentation import segment_cell
from coordinate_system import deblur_ch2, normalize_ch2, get_boundary, boundary2centerline, estimate_r, volume_elements, fluorescence_density, quantiles, average_r_quantile, average_s_quantile
import matplotlib.pyplot as plt

# Load cell
directory = 'data/extracted_cells/'
cells_early = []
cells_late = []
for f in sorted(os.listdir(directory))[:100]:
    cell = np.load(directory+f).astype(np.float64)
    cells_early.append(cell)
for f in sorted(os.listdir(directory))[-100:]:
    cell = np.load(directory+f).astype(np.float64)
    cells_late.append(cell)

# Constants
pm = 0.11

# Load model
model = tf.keras.models.load_model('segmnet')

r_s_early = []
s_s_early = []
for cell in cells_early:
    # Obtain segmentation
    pixelated_mask = segment_cell(cell[0], model)

    # Normalize ch2
    cell_fl = normalize_ch2(cell[1], pixelated_mask)

    # Deblur channel 2
    cell_fl = deblur_ch2(cell_fl, 4)

    # Get boundary
    boundary = get_boundary(pixelated_mask)

    # Get centerline
    centerline = boundary2centerline(boundary)

    # Estimate r
    r = estimate_r(boundary, centerline, pm)

    # Volume elements
    ve = volume_elements(pixelated_mask, centerline, r, pm)
#    x, y = zip(*np.argwhere(pixelated_mask==0))
#    cell_fl[x,y] = np.nan

    # Fluorescence density
    cell_fl = fluorescence_density(cell_fl, ve)

    # Quantiles
    qs = quantiles(cell_fl, 4)

    # Average r
    rq = []
    sq = []
    for quantile in qs:
        rq.append(average_r_quantile(cell_fl, quantile, centerline, r, pm))
        sq.append(average_s_quantile(cell_fl, quantile, centerline))

    r_s_early.append(rq)
    s_s_early.append(sq)

r_s_late = []
s_s_late = []
for cell in cells_late:
    # Obtain segmentation
    pixelated_mask = segment_cell(cell[0], model)

    # Normalize ch2
    cell_fl = normalize_ch2(cell[1], pixelated_mask)

    # Deblur channel 2
    cell_fl = deblur_ch2(cell_fl, 4)

    # Get boundary
    boundary = get_boundary(pixelated_mask)

    # Get centerline
    centerline = boundary2centerline(boundary)

    # Estimate r
    r = estimate_r(boundary, centerline, pm)

    # Volume elements
    ve = volume_elements(pixelated_mask, centerline, r, pm)

    # Fluorescence density
    cell_fl = fluorescence_density(cell_fl, ve)
#    x, y = zip(*np.argwhere(pixelated_mask==0))
#    cell_fl[x,y] = np.nan

    # Quantiles
    qs = quantiles(cell_fl, 4)

    # Average r
    rq = []
    sq = []
    for quantile in qs:
        rq.append(average_r_quantile(cell_fl, quantile, centerline, r, pm))
        sq.append(average_s_quantile(cell_fl, quantile, centerline))

    r_s_late.append(rq)
    s_s_late.append(sq)

r_s_late = np.array(r_s_late)
r_s_early = np.array(r_s_early)
s_s_late = np.array(s_s_late)
s_s_early = np.array(s_s_early)
print("Begining of experiment: %f" % np.mean(r_s_early[:, -1]))
print("End of experiment: %f" % np.mean(r_s_late[:, -1]))
bins = np.arange(0, 1.05, 0.05)
for i in range(4):
    plt.hist(r_s_early[:, i], bins=bins, alpha=0.6)
    plt.hist(r_s_late[:, i], bins=bins, alpha=0.6)
    plt.show()
for i in range(4):
    plt.hist(s_s_early[:, i], bins=bins, alpha=0.6)
    plt.hist(s_s_late[:, i], bins=bins, alpha=0.6)
    plt.show()


