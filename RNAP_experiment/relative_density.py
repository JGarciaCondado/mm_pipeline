import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from segmentation import segment_cell
from coordinate_system import deblur_ch2, normalize_ch2, get_boundary, boundary2centerline, estimate_r, volume_elements, fluorescence_density, quantiles, r_quantile, s_quantile, extendcenterline
import matplotlib.pyplot as plt
from matplotlib import rc
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams.update({'font.size': 16})
rc('text', usetex=True)

# Load cell
directory = 'data/bigexp/curated/'
cells = []
for f in os.listdir(directory):
    if f[-3:] == 'npy':
        cell = np.load(directory+f, allow_pickle=True).astype(np.float64)
        cells.append(cell)

# Constants
pm = 0.11

# Load model
model = tf.keras.models.load_model('segmnet')

rq1, rq2, rq3, rq4 = [], [], [], []
rqs = [rq1, rq2, rq3, rq4]
rqsu = [rq1, rq2, rq3, rq4]
sq1, sq2, sq3, sq4 = [], [], [], []
sqs = [sq1, sq2, sq3, sq4]
sqsu = [sq1, sq2, sq3, sq4]
for cell in cells:
    # Obtain segmentation
    pixelated_mask = segment_cell(cell[0], model, pad_flag=True)

    # Normalize ch2
    cell_fl = normalize_ch2(cell[1], pixelated_mask)
    cell_u = normalize_ch2(cell[0], pixelated_mask)
    x, y = zip(*np.argwhere(pixelated_mask==0))
    cell_fl[x,y] = np.nan
    cell_u[x,y] = np.nan

    # Get boundary
    boundary = get_boundary(pixelated_mask)

    # Get centerline
    centerline = boundary2centerline(boundary)

    # Estimate r
    r = estimate_r(boundary, centerline, pm)

    qs = quantiles(cell_fl, 4)
    qsu = quantiles(cell_u, 4)

    for q, qu, rq, rqu, sq, squ, i in zip(qs, qsu, rqs, rqsu, sqs, sqsu, range(4)):
        rqs[i] = np.concatenate((rq, r_quantile(cell_fl, q, centerline, r, pm)))
        rqsu[i] = np.concatenate((rqu, r_quantile(cell_u, qu, centerline, r, pm)))
        sqs[i] = np.concatenate((sq, s_quantile(cell_fl, q, centerline)))
        sqsu[i] = np.concatenate((squ, s_quantile(cell_u, qu, centerline)))

rqs = np.array(rqs)
rqsu = np.array(rqsu)
sqs = np.array(sqs)
sqsu = np.array(sqsu)
bins = np.arange(-1.00, 1.05, 0.05)
names = ['Bottom Quantile', 'Top Quantile']
for j in range(4):
    plt.subplot(2,2,j+1)
    plt.title(names[j%2])
    if j < 2:
        uniform, be = np.histogram(sqsu[j*3], bins)
        rnap, be = np.histogram(sqs[j*3], bins)
    else:
        uniform, be = np.histogram(rqsu[j%2*3], bins)
        rnap, be = np.histogram(rqs[j%2*3], bins)
    with np.errstate(divide='ignore'):
        result = rnap / uniform
        result[uniform == 0] = 0
    plt.hist(be[:-1], be, weights=result)
    if j < 2:
        plt.xlabel(r'$s$', fontsize=24)
    else:
        plt.xlabel(r'$\rho$', fontsize=24)
    plt.ylabel(r'Relative density')
plt.show()

