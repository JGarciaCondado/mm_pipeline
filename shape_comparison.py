import sys
sys.path.append('../')

from contour import contour
import numpy as np
import matplotlib.pyplot as plt
import efd
from bacteria_model import bacteria_spline
from shapely.geometry import Polygon

# Loda image
im = np.load('examples/test_bacteria.npy')

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
contour_smoothed = Polygon(contour.smoothed_contour)
contour_act = Polygon(contour.active_contour)
contour_px = Polygon(contour.pixelated_contour)
bacteria_contour = Polygon(contour.boundary).buffer(0)
dif = bacteria_contour.symmetric_difference(contour_smoothed)
plt.plot(*contour_smoothed.exterior.xy, label="smoothed")
contour_o = contour_smoothed.buffer(-.55)
plt.plot(*contour_o.exterior.xy, label = "optimal")
contour_o = contour_smoothed.buffer(-.3)
plt.plot(*contour_o.exterior.xy, label = "contraction")
contour_o = contour_smoothed.buffer(-1.0)
plt.plot(*contour_o.exterior.xy, label = "inner outline")
plt.plot(*bacteria_contour.exterior.xy, label = "ground truth")
plt.axis('scaled')
plt.title('Boundary comparison')
plt.legend(fontsize=6)
plt.show()
