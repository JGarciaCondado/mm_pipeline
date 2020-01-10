import os
import numpy as np
import sys
sys.path.append('../')
from molyso.molyso.generic.otsu import threshold_otsu
import matplotlib.pyplot as plt

directory_gt = "../dataset/GT"
directory_data = "../dataset/data"
files = os.listdir(directory_gt)
im_gt_stack = np.empty(shape=(len(files), 60, 26))
im_stack = np.empty(shape=(len(files), 60, 26))

for i, f in enumerate(files):
    im_gt = np.load(directory_gt + "/" + f)
    im_gt_stack[i, :, :] = im_gt
    im = np.load(directory_data + "/im_" +f[6:])
    im_stack[i, :, :] = im

last_error = 1.0
current_error = 0
total = 0
bias = 1.0
d_error = 1.0
step = 0.1
sign = 1.0
while (abs(d_error)>0.001):
    if d_error > 0:
        bias -= step*sign
    else:
        step = step/2
        sign = sign*-1.0
        bias -= step*sign
    current_error = 0
    total = 0
    for i, im in enumerate(im_stack):
        rm_indices = np.where(im==0.0)[0]
        im = np.delete(im, rm_indices, axis=0)
        binary_image = im > (threshold_otsu(im)*bias)
        binary_truth = np.delete(im_gt_stack[i, :, :] == 1.0, rm_indices, axis=0)
        total += binary_image.shape[0]*binary_image.shape[1]
        current_error += np.sum(binary_image != binary_truth)
    d_error = last_error - current_error/total
    # if negative change and below wanted value previous value was better
    if abs(d_error) < 0.001 and d_error<0.0:
        bias += step*sign
        current_error = last_error*total
    last_error = current_error/total
print("Bias : %s" % bias)
error = current_error/total
print("Error: %s" % error)

