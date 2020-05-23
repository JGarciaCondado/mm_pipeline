import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from segmentation import segment_cell, display, boundary_from_pixelated_mask, smooth_boundary, display_boundary
from molyso.generic.otsu import threshold_otsu
from matplotlib_scalebar.scalebar import ScaleBar
import os

cells = []
directory = 'data/extracted_cells/'
for f in os.listdir(directory)[:4]:
    cell = np.load(directory+f)[0]
    cells.append(cell)

model = tf.keras.models.load_model('segmnet')

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
axes = [ax1, ax2, ax3, ax4]
scalebar = ScaleBar(0.11, 'um', frameon=False, color='w', location=2) # 1 pixel = 0.2 meter
plt.gca().add_artist(scalebar)
title = ['Pixelated Boundary', 'Smoothed Boundary']
colors = ['r', 'g']
for i, cell in enumerate(cells):
    axes[i].imshow(cell)
    axes[i].axis('off')
    pixelated_mask = segment_cell(cell, model)
    boundary = boundary_from_pixelated_mask(pixelated_mask)
    smoothed_boundary = smooth_boundary(boundary, 5)
    boundaries = [boundary]#, smoothed_boundary]
    for j in range(len(boundaries)):
        axes[i].plot(boundaries[j][:,0], boundaries[j][:,1], colors[j], label=title[j], linewidth=1.5)
plt.show()

#Pixelation by otsu
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
axes = [ax1, ax2, ax3, ax4]
title = ['Pixelated Boundary', 'Smoothed Boundary']
colors = ['r', 'g']
scalebar = ScaleBar(0.11, 'um', frameon=False, color='w', location=2) # 1 pixel = 0.2 meter
plt.gca().add_artist(scalebar)
for i, cell in enumerate(cells):
    axes[i].imshow(cell)
    axes[i].axis('off')
    pixelated_mask = cell > threshold_otsu(cell)
    boundary = boundary_from_pixelated_mask(pixelated_mask)
    smoothed_boundary = smooth_boundary(boundary, 5)
    boundaries = [boundary]#, smoothed_boundary]
    for j in range(len(boundaries)):
        axes[i].plot(boundaries[j][:,0], boundaries[j][:,1], colors[j], label=title[j], linewidth=1.5)
plt.show()
