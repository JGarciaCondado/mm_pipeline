import numpy as np
import os
import sys
sys.path.append('../')
from molyso.generic.otsu import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from skimage.segmentation import active_contour
from bacteria_model import bacteria_spline
import re

directory = "../dataset/data/"
files = os.listdir(directory)[:1]
im_stack = np.empty(shape=(len(files), 60, 26))
r, l, R, theta = [float(s) for s in re.findall(r"\d+\.\d+", files[0])]
# Ball park values
ex_wv = 0.8  # emmitted wavelength by microscope for excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
n_b = 0

def spline_fn_curvature(x):
    return np.sqrt(R**2 - (x-l/2)**2) - np.sqrt(R**2-l**2/4)

bacteria = bacteria_spline(r, l, 0.01, spline_fn_curvature, theta, ex_wv, em_wv, n_b)

for i, f in enumerate(files):
    im = np.load(directory + f)
    im_stack[i, :, :] = im

def initial_contour(im):
    binary_image = im > (threshold_otsu(im)*0.8)
    rows, columns = np.where(binary_image == True)
    prev_r = 0
    coordinates_l = []
    coordinates_r = []
    for i, r in enumerate(rows):
        if prev_r != r:
            coordinates_l.append([columns[i], r])
            coordinates_r.append([columns[i-1], prev_r])
            prev_r = r
    coordinates_r.pop(0) # pop initial useless value
    coordinates_r.append([columns[-1], rows[-1]]) # add last value missing
    coordinates_r.reverse() # revers to add to l
    coordinates = coordinates_l + coordinates_r
    coordinates.append(coordinates[0]) # duplicate start point
    return coordinates

m = 40
pixel_size = 4.4
padding = 2

def transform_vertices(verts, bacteria):
    verts = verts - bacteria.min[:-1] # move to get non-zero values
    verts = verts*m #magnification
    verts = verts / pixel_size #scaling by size of pixels
    verts = verts + padding # add padding
    verts[:,[0, 1]] = verts[:,[1, 0]] #make horizontal
    return verts

for i, im in enumerate(im_stack):
    rm_indices = np.where(im==0.0)[0]
    im = np.delete(im, rm_indices, axis=0)
    init_contour = initial_contour(im)
    fig, ax = plt.subplots()
    codes = [Path.MOVETO] + [Path.LINETO] * (len(init_contour) - 2) + [Path.CLOSEPOLY]
    path = Path(init_contour, codes)
    patch = patches.PathPatch(path, fill=False, lw=2, ec = 'orange')
    ax.add_patch(patch)
    init_contour.pop()
    init_contour = [val for val in init_contour for _ in range(10)]
    act_contour = active_contour(im, np.array(init_contour))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(act_contour) - 2) + [Path.CLOSEPOLY]
    path = Path(act_contour, codes)
    patch = patches.PathPatch(path, fill=False, lw=2, ec = 'red')
    ax.add_patch(patch)
    gt_contour = bacteria.boundary[:, :-1]
    gt_contour = transform_vertices(gt_contour, bacteria)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(gt_contour) - 1)
    path = Path(gt_contour, codes)
    patch = patches.PathPatch(path, fill=False, lw=2, ec = 'green')
    ax.add_patch(patch)
    plt.imshow(im)
    plt.show()
