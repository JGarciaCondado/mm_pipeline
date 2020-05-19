import tensorflow as tf
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from segmentation import segment_cell, display, boundary_from_pixelated_mask, smooth_boundary, display_boundary

cell = np.load("cell_examples/Cell_test_1.npy")
# Remove zeros
rm_indices = np.where(cell==0.0)[0]
cell = np.delete(cell, rm_indices, axis=0)

model = tf.keras.models.load_model('../saved_model/segmentation')
pixelated_mask = segment_cell(cell, model)
boundary = boundary_from_pixelated_mask(pixelated_mask)

for i in range(10):
    plt.subplot(2,5,i+1)
    if i ==0:
        scalebar = ScaleBar(0.11, 'um', frameon=False, color='w', location=2) # 1 pixel = 0.2 meter
        plt.gca().add_artist(scalebar)
    plt.title('nº of EFDs: {}'.format(i+1))
    plt.imshow(cell)
    smoothed_boundary = smooth_boundary(boundary, i+1)
    plt.plot(smoothed_boundary[:,0], smoothed_boundary[:,1], 'r')
    plt.axis('off')
plt.show()
