import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

directory_data = "../dataset/data"
files = os.listdir(directory_data)[:16]
N_half = 4
n_rows = 4
fig, ax = plt.subplots()
# Create red color map for image show
colors = [(0, 0, 0), (1, 0, 0)]

for i, f in enumerate(files):
    im = np.load(directory_data + "/" + f)
    ax = plt.subplot2grid((n_rows, N_half), (i // N_half, i % N_half))
    cm = LinearSegmentedColormap.from_list('test', colors, N=np.amax(im))
    ax.imshow(im, cmap=cm, origin="lower")
plt.show()
