from contour import contour
import os
import numpy as np
import re
from bacteria_model import bacteria_spline

directory = "dataset/data/"
files = os.listdir(directory)[:1]
im_stack = np.empty(shape=(len(files), 60, 26))
r, l, R, theta = [float(s) for s in re.findall(r"\d+\.\d+", files[0])]
# Ball park values
ex_wv = 0.8  # emmitted wavelength by microscope for excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
n_b = 0

m = 40
pixel_size = 4.4
padding = 2

def spline_fn_curvature(x):
    return np.sqrt(R**2 - (x-l/2)**2) - np.sqrt(R**2-l**2/4)

bacteria = bacteria_spline(r, l, 0.01, spline_fn_curvature, theta, ex_wv, em_wv, n_b)

for i, f in enumerate(files):
    im = np.load(directory + f)
    im_stack[i, :, :] = im
rm_indices = np.where(im==0.0)[0]
im = np.delete(im, rm_indices, axis=0)
contour = contour(im, 0.8, bacteria, m, pixel_size, padding)
contour.show_pixelated_contour()
contour.show_smoothed_contour()
contour.show_active_contour()
contour.show_contours()
