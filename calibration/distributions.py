import numpy as np
import os
import matplotlib.pyplot as plt
from tifffile import imread
from extraction import extract_params
from scipy.stats import norm

directory = "Data"
params = []
for f in os.listdir(directory):
    if f[-4:] != ".tif":
        continue
    im=imread(directory+'/'+f)
    try:
        l, R, theta = extract_params(im, 40, 4.4)
        params.append([l, R, theta])
    except:
        pass

params = np.array(params)
mu_l, std_l = norm.fit(params[:, 0])
bins = np.arange(0, 6, 0.25)
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
