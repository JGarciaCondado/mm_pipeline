import numpy as np
import os
import matplotlib.pyplot as plt
from tifffile import imread
from extraction import extract_params

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
bins = np.arange(0, 6, 0.25)
plt.hist(params[:, 0], bins)
plt.show()
params[:, 1] = [abs(params[i,1]/params[i,0]) if abs(params[i,1]/params[i,0]) < 10 else 10 for i in range(len(params[:,1]))]
bins = np.arange(0, 10.5, 0.5)
plt.hist(params[:, 1], bins)
plt.show()
plt.hist(params[:, 2])
plt.show()
