import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import sys
sys.path.append('../')
from segmentation import segment_cell, boundary_from_pixelated_mask, smooth_boundary, display, smooth_fd_boundary
from contour import contour_real, boundary
from shapely.geometry import Polygon, LinearRing, Point

cells = np.load('../dataset/im_stack.npy')[:1000]
params = np.load('../dataset/params.npy', allow_pickle=True)[:1000]
model = tf.keras.models.load_model('../saved_model/segmentation')

def distance(contour, coords):
    ring = LinearRing(list(contour.exterior.coords))
    error = []
    for coord in coords:
        point = Point(coord)
        distance = contour.distance(point)
        error.append(distance)
    return np.mean(error)

m = 40
pixel_size = 4.4
pix_error =[]
efd_error = []
fd_error = []
pix_error_d =[]
efd_error_d = []
fd_error_d = []
ac_error = []
ml_error =[]
smoothed_efd_ml_error = []
smoothed_fd_ml_error = []
for im, param in zip(cells, params):
    contour = contour_real(im, 1.0)
    pixelated_boundary = Polygon(contour.pixelated_contour).buffer(0.0)
    efd_boundary = Polygon(contour.smoothed_contour).buffer(0.0)
    fd_boundary = Polygon(contour.fd_contour).buffer(0.0)
    ac_boundary = Polygon(contour.active_contour).buffer(0.0)
    r,l,R,theta,centroid = param
    bound = boundary(r,l,R,theta)
    cell_bound = bound.get_boundary(m, pixel_size, centroid)
    cell_bound = Polygon(cell_bound).buffer(0)
    pix_error.append(cell_bound.symmetric_difference(pixelated_boundary).area/cell_bound.area)
    efd_error.append(cell_bound.symmetric_difference(efd_boundary).area/cell_bound.area)
    fd_error.append(cell_bound.symmetric_difference(fd_boundary).area/cell_bound.area)
    ac_error.append(cell_bound.symmetric_difference(ac_boundary).area/cell_bound.area)
    pixelated_mask = segment_cell(im, model, pad_flag=False)
    bound = boundary_from_pixelated_mask(pixelated_mask)
    pix_bound_ml = Polygon(bound).buffer(0.0)
    smoothed_boundary = Polygon(smooth_boundary(bound, 5)).buffer(0.0)
    smoothed_fd_boundary = Polygon(smooth_fd_boundary(bound, 5)).buffer(0.0)
    ml_error.append(cell_bound.symmetric_difference(pix_bound_ml).area/cell_bound.area)
    smoothed_efd_ml_error.append(cell_bound.symmetric_difference(smoothed_boundary).area/cell_bound.area)
    smoothed_fd_ml_error.append(cell_bound.symmetric_difference(smoothed_fd_boundary).area/cell_bound.area)
    # TODO check that boundaries are not multipolygon
    try:
        pix_error_d.append(distance(cell_bound, list(pix_bound_ml.exterior.coords)))
        efd_error_d.append(distance(cell_bound, list(smoothed_boundary.exterior.coords)))
        fd_error_d.append(distance(cell_bound, list(smoothed_fd_boundary.exterior.coords)))
    except:
        pass
print(np.mean(pix_error))
print(np.mean(efd_error))
print(np.mean(fd_error))
print(np.mean(pix_error_d)*0.11)
print(np.mean(efd_error_d)*0.11)
print(np.mean(fd_error_d)*0.11)
print(np.mean(ml_error))
print(np.mean(smoothed_fd_ml_error))
print(np.mean(smoothed_efd_ml_error))

#Max
print('Max')
print(np.max(pix_error))
print(np.max(efd_error))
print(np.max(fd_error))
print(np.max(ml_error))
print(np.max(smoothed_fd_ml_error))
print(np.max(smoothed_efd_ml_error))

#Min
print('Min')
print(np.min(pix_error))
print(np.min(efd_error))
print(np.min(fd_error))
print(np.min(ml_error))
print(np.min(smoothed_fd_ml_error))
print(np.min(smoothed_efd_ml_error))
