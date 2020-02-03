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

total_error = 0.0
total_area = 0.0
error = []

for i in tqdm(range(stack_len)):
    r, l, R, theta = parameters_stack[i,:]
    im = im_stack[i,:,:]
    bacteria = bacteria_spline(r, l, 0.01, spline_fn_curvature, theta, ex_wv, em_wv, n_b)

    # Remove blank padding
    rm_indices = np.where(im==0.0)[0]
    im = np.delete(im, rm_indices, axis=0)
    icontour = contour(im, 0.8, bacteria, m, pixel_size, padding, 7)
    contour_smoothed = Polygon(icontour.smoothed_contour).buffer(0)
    bacteria_contour = Polygon(icontour.boundary).buffer(0)
#    erosion = -1.175*r+0.8725
    erosion = 0.25
    contour_c = contour_smoothed.buffer(-erosion)
    dif = bacteria_contour.symmetric_difference(contour_c)
    error.append(dif.area/bacteria_contour.area)
    total_error += dif.area
    total_area += bacteria_contour.area

print("Total error: {}".format(total_error/total_area))
print("Average error:{} ".format(np.mean(error)))
print("Standard deviation: {}".format(np.std(error)))
bin_edges = np.arange(0.0, np.max(error)+0.01, 0.01)
plt.hist(error, bins = bin_edges)
plt.show()
