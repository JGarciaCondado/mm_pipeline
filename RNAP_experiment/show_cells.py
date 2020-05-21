import numpy as np
import matplotlib.pyplot as plt
import re
import os

directory = 'data/extracted_cells/'
for f in os.listdir(directory)[:10]:
    im = np.load(directory+f)
    im_seg = im[0]
    im_fl = im[1]
    plt.subplot(1,2,1)
    plt.title('Channel 1')
    plt.imshow(im_seg)
    plt.subplot(1,2,2)
    plt.title('Channel 2')
    plt.imshow(im_fl)
    plt.show()
