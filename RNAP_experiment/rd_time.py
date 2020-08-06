import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from segmentation import segment_cell
from coordinate_system import deblur_ch2, normalize_ch2, get_boundary, boundary2centerline, estimate_r, volume_elements, fluorescence_density, quantiles, r_quantile, s_quantile, extendcenterline
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from matplotlib import rc
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams.update({'font.size': 16})
rc('text', usetex=True)

# Load cell
start = 'data/bigexp/curated/'
end = 'data/endexp/curated/'
start_cells = []
end_cells = []
for f in os.listdir(start):
    if f[-3:] == 'npy':
        cell= np.load(start+f, allow_pickle=True).astype(np.float64)
        start_cells.append(cell)

for f in os.listdir(end):
    if f[-3:] == 'npy':
        cell= np.load(end+f, allow_pickle=True).astype(np.float64)
        end_cells.append(cell)

# Constants
pm = 0.11

# Load model
model = tf.keras.models.load_model('segmnet')

rq1, rq2, rq3, rq4 = [], [], [], []
rqs1 = [rq1, rq2, rq3, rq4]
rqs2 = [rq1, rq2, rq3, rq4]
sq1, sq2, sq3, sq4 = [], [], [], []
sqs1 = [sq1, sq2, sq3, sq4]
sqs2 = [sq1, sq2, sq3, sq4]
for cell in start_cells:
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

    for q, rq, sq, i in zip(qs, rqs1, sqs1, range(4)):
        rqs1[i] = np.concatenate((rq, r_quantile(cell_fl, q, centerline, r, pm)))
        sqs1[i] = np.concatenate((sq, s_quantile(cell_fl, q, centerline)))

for cell in end_cells:
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

    for q, rq, sq, i in zip(qs, rqs2, sqs2, range(4)):
        rqs2[i] = np.concatenate((rq, r_quantile(cell_fl, q, centerline, r, pm)))
        sqs2[i] = np.concatenate((sq, s_quantile(cell_fl, q, centerline)))

rqs1 = np.array(rqs1)
rqs2 = np.array(rqs2)
sqs1 = np.array(sqs1)
sqs2 = np.array(sqs2)
bins = np.arange(-1.00, 1.05, 0.05)
names = ['Bottom Quantile', 'Top Quantile']
for j in range(4):
    plt.subplot(2,2,j+1)
    plt.title(names[j%2])
    if j < 2:
        start, be = np.histogram(sqs1[j*3], bins, density = True)
        end, be = np.histogram(sqs2[j*3], bins, density = True)
    else:
        start, be = np.histogram(rqs1[j%2*3], bins, density = True)
        end, be = np.histogram(rqs2[j%2*3], bins, density = True)
    with np.errstate(divide='ignore'):
        result = end / start
        result[start == 0] = 0
    plt.hist(be[:-1], be, weights=result)
    if j < 2:
        plt.xlabel(r'$s$', fontsize=24)
    else:
        plt.xlabel(r'$\rho$', fontsize=24)
    plt.ylabel(r'Relative density')
plt.show()

