import numpy as np
import matplotlib.pyplot as plt
import os

start_exp = 'data/bigexp/'
end_exp = 'data/endexp/'

exps = ['start', 'end']
dir_exps = [start_exp, end_exp]

def ask_input():
     x = input()
     if x == 'y' or x == 'n':
         return x
     else:
         return ask_input()

for exp, dir_exp in zip(exps, dir_exps):
    i = 1
    for f in os.listdir(dir_exp+'raw/'):
        if f[-3:] == 'npy':
            cell = np.load(dir_exp+'raw/'+f, allow_pickle=True).astype(np.float64)
            plt.imshow(cell[0])
            plt.pause(0.001)
            x = ask_input()
            if x == 'y':
                print('accept')
                np.save(dir_exp+'curated/'+f, cell)
            elif x == 'n':
                print('reject')
            plt.clf()

