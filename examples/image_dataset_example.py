import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar

ims = np.load('../dataset/im_stack.npy')[152:168]
N_half = 4
n_rows = 4
fig, ax = plt.subplots()
# Create red color map for image show
colors = [(0, 0, 0), (1, 0, 0)]

for i, im in enumerate(ims):
    if i==1:
        scalebar = ScaleBar(0.11, 'um', frameon=False, color='w') # 1 pixel = 0.2 meter
        plt.gca().add_artist(scalebar)
    ax = plt.subplot2grid((n_rows, N_half), (i // N_half, i % N_half))
    cm = LinearSegmentedColormap.from_list('test', colors, N=np.amax(im))
    ax.imshow(im, cmap=cm, vmin=0, origin="lower")
    ax.axis('off')
plt.show()
