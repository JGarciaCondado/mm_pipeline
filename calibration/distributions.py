import numpy as np
import os
import matplotlib.pyplot as plt
from tifffile import imread
from extraction import extract_all_params, extract_params
from scipy.stats import norm
from scipy.interpolate import splrep, sproot, splev
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})

directory = "Data"
params = []
for f in os.listdir(directory):
    if f[-4:] != ".tif":
        continue
    im=imread(directory+'/'+f)
    rm_indices = np.where(im==0.0)[0]
    im = np.delete(im, rm_indices, axis=0)
    try:
        l, R, theta, centroid = extract_all_params(im, 40, 4.4)
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
plt.xlabel('centroid x')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_x, std_x)
plt.title(title)
plt.show()
mu_y, std_y = norm.fit(params[:, 4])
plt.hist(params[:, 4], density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_y, std_y)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('centroid y')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_y, std_y)
plt.title(title)
plt.show()

mu_l, std_l = norm.fit(params[:, 0])
bins = np.arange(0, 6, 0.25)
hist, bin_edges = np.histogram(params[:, 0], bins=bins, normed=True)
spl = splrep(np.arange(0.125, 5.8, 0.25), hist)
xim = np.arange(0, 6, 0.01)
yim = splev(xim, spl)
yim = [y if y>0.0 else 0.0 for y in yim]
#plt.plot(xim, yim)
plt.hist(sorted(params[:, 0])[1:], bins, density=True, label='Experimental PDF')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_l, std_l)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian fit')
plt.xlabel(r'l ($\mu$m)')
plt.ylabel(r'p(x)')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_l, std_l)
plt.title(title)
plt.legend()
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
