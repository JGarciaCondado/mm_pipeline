import numpy as np
import matplotlib.pyplot as plt
import re
import os

directory = 'data/extracted_cells/'
for f in os.listdir(directory)[:10]:
    im_type = f.split("_")[2]
    if im_type == 'c':
        im = np.load(directory+f)
        f = f[:13] + 'y' + f[14:]
        im_fl = np.load(directory+f)
    else:
        im_fl = np.load(directory+f)
        f = f[:13] + 'c' + f[14:]
        im = np.load(directory+f)
    plt.subplot(1,2,1)
    plt.title('Channel 1')
    plt.imshow(im)
    plt.subplot(1,2,2)
    plt.title('Channel 2')
    plt.imshow(im_fl)
    plt.show()
