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

contraction = np.arange(0.0, 1.1, 0.1)
total_error = [0.0]*len(contraction)
total_error_ac = 0.0
total_error_px = 0.0
total_area = 0.0

for i in tqdm(range(stack_len)):
    r, l, R, theta = parameters_stack[i,:]
    im = im_stack[i,:,:]
    bacteria = bacteria_spline(r, l, 0.01, spline_fn_curvature, theta, ex_wv, em_wv, n_b)

    # Remove blank padding
    rm_indices = np.where(im==0.0)[0]
    im = np.delete(im, rm_indices, axis=0)
    icontour = contour(im, 0.8, bacteria, m, pixel_size, padding, 7)
    contour_smoothed = Polygon(icontour.pixelated_contour).buffer(0)
    bacteria_contour = Polygon(icontour.boundary).buffer(0)
    for j, erosion in enumerate(contraction):
        contour_c = contour_smoothed.buffer(-erosion)
        dif = bacteria_contour.symmetric_difference(contour_c)
        total_error[j] += dif.area
    total_area += bacteria_contour.area
    contour_px = Polygon(icontour.pixelated_contour)
    total_error_px += bacteria_contour.symmetric_difference(contour_px).area

error = [e/total_area for e in total_error]
print(contraction[np.argwhere(error == np.min(error))], np.min(error))
plt.plot(contraction, error)
plt.title("Contraction effect on error")
plt.xlabel("Contraction parameter")
plt.ylabel("Average error")
plt.show()
