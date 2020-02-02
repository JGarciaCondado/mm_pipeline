import os
import numpy as np
import re
from tqdm import tqdm
from bacteria_model import bacteria_spline
from shapely.geometry import Polygon
from contour import contour
import matplotlib.pyplot as plt

directory_data = "dataset/data"
files = os.listdir(directory_data)
# get only non_flipped files
original_f = sorted(files)[:9200:4]
stack_len = len(original_f)
im_stack = np.empty(shape=(len(original_f), 60, 26))
parameters_stack = np.empty(shape = (len(original_f), 4))

for i, f in tqdm(enumerate(original_f)):
    im_stack[i,:,:] = np.load(directory_data+"/"+f)
    parameters_stack[i,:] = re.findall(r"[-+]?\d*\.\d+", f)

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

f_order = range(1, 22, 2)
total_error = [0.0]*len(f_order)
total_area = 0.0

for i in tqdm(range(stack_len)):
    r, l, R, theta = parameters_stack[i,:]
    im = im_stack[i,:,:]
    bacteria = bacteria_spline(r, l, 0.01, spline_fn_curvature, theta, ex_wv, em_wv, n_b)

    # Remove blank padding
    rm_indices = np.where(im==0.0)[0]
    im = np.delete(im, rm_indices, axis=0)
    icontour = contour(im, 0.8, bacteria, m, pixel_size, padding)
    bacteria_contour = Polygon(icontour.boundary).buffer(0)
    for j, f in enumerate(f_order):
        icontour.re_smooth(f)
        contour_smoothed = Polygon(icontour.smoothed_contour)
        contour_smoothed = contour_smoothed.buffer(0)
        dif = bacteria_contour.symmetric_difference(contour_smoothed)
        total_error[j] += dif.area
    total_area += bacteria_contour.area

error = [e/total_area for e in total_error]
plt.plot(f_order, error)
plt.title("Nº of Fourier Descriptors effect on error")
plt.xlabel("Nº of Fourier Descriptors")
plt.ylabel("Total error")
plt.show()
