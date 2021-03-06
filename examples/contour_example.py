import sys
sys.path.append('../')

from contour import contour
import numpy as np
from bacteria_model import bacteria_spline

# Loda image
im = np.load('test_bacteria.npy')

# Bacteria values
r = 0.3309786038590506
l = 2.9239029503218905
R = 15.336402399051828
theta= 12.032521406278008
ex_wv = 0.8
em_wv = 0.59
n_b = 0

# Microscope values
m = 40
pixel_size = 4.4
padding = 2

# create bacteria model
def spline_fn_curvature(x):
    return np.sqrt(R**2 - (x-l/2)**2) - np.sqrt(R**2-l**2/4)

bacteria = bacteria_spline(r, l, 0.01, spline_fn_curvature, theta, ex_wv, em_wv, n_b)

# Remove blank padding
rm_indices = np.where(im==0.0)[0]
im = np.delete(im, rm_indices, axis=0)

# Display contour
contour = contour(im, 0.8, bacteria, m, pixel_size, padding)
contour.show_pixelated_contour()
contour.show_smoothed_contour()
contour.show_active_contour()
contour.show_contours()
