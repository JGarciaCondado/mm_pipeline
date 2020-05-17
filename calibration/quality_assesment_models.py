from comparison import MSE, l2_norm, MSSIM
import numpy as np
import os
import sys
sys.path.append('../')
from models import SpherocylindricalBacteria, Microscope
from tifffile import imread
from extraction import extract_all_params
import matplotlib.pyplot as plt

pixel_size = 4.4  # pixel size
NA = 0.95  # Numerical aperture
m = 40  # magnification
ex_wv = 0.8
em_wv = 0.59

#Images
directory = "Data"
quality = []
microscope = Microscope(m, NA, ex_wv, em_wv, pixel_size)
files = sorted(os.listdir(directory))
for f in files[:101]:
    if f[-4:] != ".tif":
        continue
    im=imread(directory+'/'+f)
    try:
        rm_indices = np.where(im==0.0)[0]
        im = np.delete(im, rm_indices, axis=0)
        shape = im.shape
        l, R, theta, centroid = extract_all_params(im, 40, 4.4)
        r = 0.5
        bacteria = SpherocylindricalBacteria(l-r, r, R, theta, 1700,  ex_wv, em_wv)
        synth_im = microscope.image_bacteria_cauchy(bacteria, centroid, shape, gamma=2.4)
        im = (im-np.min(im))/(np.max(im)-np.min(im))
        synth_im = (synth_im-np.min(synth_im))/(np.max(synth_im)-np.min(synth_im))
        quality.append([MSE(im, synth_im), l2_norm(im, synth_im), MSSIM(im, synth_im)])
#        plt.subplot(1,2,1)
#        plt.imshow(im)
#        plt.subplot(1,2,2)
#        plt.imshow(synth_im)
#        plt.show()
    except:
        pass
print(np.mean(quality, axis=0))
print(np.max(quality, axis=0))
print(np.min(quality, axis=0))
