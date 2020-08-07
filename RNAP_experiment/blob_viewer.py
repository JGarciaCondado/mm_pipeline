import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import blob_dog

directory = 'data/bigexp/curated/'
fig, axes = plt.subplots(4, 5)
ax = axes.ravel()
idx = 0
for f in os.listdir(directory)[:20]:
    if f[-3:] == 'npy':
        cell = np.load(directory+f, allow_pickle=True).astype(np.float64)
        cell_fl = (cell[1]-np.min(cell[1]))/(np.max(cell[1]) - np.min(cell[1]))
        blobs_dog = blob_dog(cell_fl, max_sigma=5, threshold=0.1)
        blobs_dog[:, 2] = blobs_dog[:, 2] * 2**0.5
        ax[idx].imshow(cell_fl)
        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()
        idx += 1
plt.tight_layout()
plt.show()
