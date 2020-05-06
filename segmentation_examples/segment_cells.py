import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from segmentation import segment_cell, display, boundary_from_pixelated_mask, smooth_boundary, display_boundary

cells = []
for i in range(4):
    cell = np.load("cell_examples/Cell_test_{}.npy".format(i))
    # Remove zeros
    rm_indices = np.where(cell==0.0)[0]
    cell = np.delete(cell, rm_indices, axis=0)
    cells.append(cell)

model = tf.keras.models.load_model('../saved_model/segmentation')

fig = plt.figure(figsize=(15, 15))
title = ['Pixelated Boundary', 'Smoothed Boundary']
colors = ['r', 'g']
for i, cell in enumerate(cells):
    ax = plt.subplot(2,2,i+1)
    plt.imshow(cell)
    pixelated_mask = segment_cell(cell, model)
    boundary = boundary_from_pixelated_mask(pixelated_mask)
    smoothed_boundary = smooth_boundary(boundary, 5)
    boundaries = [boundary, smoothed_boundary]
    for i in range(len(boundaries)):
        plt.plot(boundaries[i][:,0], boundaries[i][:,1], colors[i], label=title[i])
    plt.legend()
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=5)
plt.show()
