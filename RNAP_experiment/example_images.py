import numpy as np
import matplotlib.pyplot as plt
import os

start_exp = 'data/bigexp/curated/'
end_exp = 'data/endexp/curated/'

exps = ['start', 'end']
dir_exps = [start_exp, end_exp]

for exp, dir_exp in zip(exps, dir_exps):
    i = 1
    plt.suptitle("Experiment: {}".format(exp))
    for f in os.listdir(dir_exp)[:25]:
        if f[-3:] == 'npy':
            cell = np.load(dir_exp+f, allow_pickle=True).astype(np.float64)
            plt.subplot(5,10,i)
            plt.title('Ch 1')
            plt.imshow(cell[0], aspect='auto')
            plt.axis('off')
            i += 1
            plt.subplot(5,10,i)
            plt.title('Ch 2')
            plt.imshow(cell[1], aspect='auto')
            plt.axis('off')
            i += 1
    plt.show()
