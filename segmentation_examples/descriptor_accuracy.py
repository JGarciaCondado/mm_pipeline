import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from segmentation import segment_cell, boundary_from_pixelated_mask, smooth_boundary, display
from shapely.geometry import Polygon
from contour import boundary
plt.rcParams.update({'font.size': 14})

cells = np.load('../dataset/im_stack.npy')[:100]
params = np.load('../dataset/params.npy', allow_pickle=True)[:100]

model = tf.keras.models.load_model('../saved_model/segmentation')

cell_polygons = []
for cell in cells:
    pixelated_mask = segment_cell(cell, model, pad_flag=False)
    bound = boundary_from_pixelated_mask(pixelated_mask)
    cell_descriptors = []
    for i in range(10):
        # TODO make this segmenetation all at once
        smoothed_boundary = smooth_boundary(bound, i+1)
        cell_descriptors.append(Polygon(smoothed_boundary).buffer(0.0))
    cell_polygons.append(cell_descriptors)

m = 40
pixel_size = 4.4
diff_polygons = []
for i, param in enumerate(params):
    r,l,R,theta,centroid = param
    bound = boundary(r,l,R,theta)
    cell_bound = bound.get_boundary(m, pixel_size, centroid)
    cell_bound = Polygon(cell_bound).buffer(0)
    diff_descriptors = []
    for poly in cell_polygons[i]:
        diff = cell_bound.symmetric_difference(poly)
        perc_diff = diff.area/cell_bound.area
        diff_descriptors.append(perc_diff)
    diff_polygons.append(diff_descriptors)

plt.plot(range(1,11), np.mean(diff_polygons, axis=0))
plt.xlabel("Number of descriptors", fontsize=12)
plt.ylabel("Average RPAE", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
