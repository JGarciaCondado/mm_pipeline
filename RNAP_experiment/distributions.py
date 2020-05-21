import numpy as np
import os
import matplotlib.pyplot as plt
from tifffile import imread
import sys
sys.path.append('../')
from calibration.extraction import extract_all_params, extract_params
from scipy.stats import norm
from scipy.interpolate import splrep, sproot, splev

directory = "data/extracted_cells/"
params = []
for f in os.listdir(directory):
    if f[-4:] != ".npy":
        continue
    im=np.load(directory+'/'+f)[0]
    try:
        l, R, theta, centroid = extract_all_params(im, 60, 6.6)
        cx = centroid[0]/im.shape[1]
        cy = centroid[1]/im.shape[0]
        params.append([l, R, theta, cx, cy])
    except:
        pass

params = np.array(params)
mu_x, std_x = norm.fit(params[:, 3])
plt.hist(params[:, 3], density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_x, std_x)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('centroid_x')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_x, std_x)
plt.title(title)
plt.show()
mu_y, std_y = norm.fit(params[:, 4])
plt.hist(params[:, 4], density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_y, std_y)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('centroid_y')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_y, std_y)
plt.title(title)
plt.show()


bins = np.arange(0, 8, 0.25)
mu_l, std_l = norm.fit(params[:, 0])
plt.hist(params[:, 0], bins, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_l, std_l)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('l (micrometers)')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_l, std_l)
plt.title(title)
plt.show()
params[:, 1] = [abs(params[i,1]/params[i,0]) if abs(params[i,1]/params[i,0]) < 10 else 10 for i in range(len(params[:,1]))]
bins = np.arange(0, 10.5, 0.5)
plt.hist(params[:, 1], bins)
plt.xlabel('ratio R/l')
plt.show()
mu_theta, std_theta = norm.fit(params[:, 2])
plt.hist(params[:, 2], density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_theta, std_theta)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('theta (º)')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_theta, std_theta)
plt.title(title)
plt.show()
