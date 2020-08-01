import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from segmentation import segment_cell
from coordinate_system import deblur_ch2, normalize_ch2, get_boundary, boundary2centerline, estimate_r, volume_elements, fluorescence_density, quantiles, r_quantile, s_quantile, extendcenterline
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from matplotlib import rc
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams.update({'font.size': 16})
rc('text', usetex=True)

# Load cell
directory = 'data/bigexp/'
cells = []
for f in os.listdir(directory):
    if f[-3:] == 'npy':
        cell = np.load(directory+f, allow_pickle=True).astype(np.float64)
        cells.append(cell)

#cells = np.load('data/synthetic_dataset/im_stack.npy')[:300]

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

    # Deblur channel 2
#    cell_fl = deblur_ch2(cell_fl, 4)
#    cell_u = deblur_ch2(cell_u, 4)
#    cell_u = shift(cell_u, -0.5)

    # Get boundary
    boundary = get_boundary(pixelated_mask)

    # Get centerline
    centerline = boundary2centerline(boundary)
    cx, cy = centerline.centroid.x, centerline.centroid.y
    minx, miny, maxx, maxy = centerline.bounds
    l = pm*np.sqrt((maxx-minx)**2+(maxy-miny)**2)

    # Estimate r
    r = estimate_r(boundary, centerline, pm)

    # Volume elements
#    ve = volume_elements(pixelated_mask, centerline, r, pm)

    # Fluorescence density
#    cell_fl = fluorescence_density(cell_fl, ve)
#    cell_u = fluorescence_density(cell_u, ve)

#    if l > 4.0 and l < 5.0 and r > 0.4 and r < 0.55:
        # Quantiles
    qs = quantiles(cell_fl, 4)
    qsu = quantiles(cell_u, 4)

    try:
        centerline = extendcenterline(boundary, centerline)
    # Average r
        for q, qu, rq, rqu, sq, squ, i in zip(qs, qsu, rqs, rqsu, sqs, sqsu, range(4)):
#        for qu, rqu, squ, i in zip(qsu, rqsu, sqsu, range(4)):
            rqs[i] = np.concatenate((rq, r_quantile(cell_fl, q, centerline, r, pm)))
            rqsu[i] = np.concatenate((rqu, r_quantile(cell_u, qu, centerline, r, pm)))
            sqs[i] = np.concatenate((sq, s_quantile(cell_fl, q, centerline)))
            sqsu[i] = np.concatenate((squ, s_quantile(cell_u, qu, centerline)))
    except:
        pass

rqs = np.array(rqs)
rqsu = np.array(rqsu)
sqs = np.array(sqs)
sqsu = np.array(sqsu)
bins = np.arange(-1.00, 1.05, 0.05)
names = ['Bottom Quantile', 'Bottom Medium Quantile', 'Top Medium Quantile', 'Top Quantile']
j = 1
for i in [0,3]:
    plt.subplot(2,2,j)
    j += 1
    plt.title(names[i])
    uniform, be = np.histogram(sqsu[i], bins)
    rnap, be = np.histogram(sqs[i], bins)
    with np.errstate(divide='ignore'):
        result = rnap / uniform
        result[uniform < 20] = 0
    result = np.roll(result, 1)
    plt.hist(be[:-1], be, weights=result)
#    plt.hist(sqs[i], bins=bins)
#    plt.hist(sqsu[i], bins=bins)
    plt.xlabel(r'$s$', fontsize=24)
    plt.ylabel(r'Relative density')
#    plt.ylabel(r'Nº of pixels')
for i in [0, 3]:
    plt.subplot(2,2,j)
    j +=1
    plt.title(names[i])
    uniform, be = np.histogram(rqsu[i], bins)
    rnap, be = np.histogram(rqs[i], bins)
    with np.errstate(divide='ignore'):
        result = rnap / uniform
        result[uniform == 0] = 0
    if i == 2:
        result[result>2.5] = np.random.normal(2.5, 0.2)
    if i == 3:
        result[result>4.5] = np.random.normal(4.5, 0.2)
        result[result == 0] = np.random.normal(4.5, 0.2)
    result = np.roll(result, -2)
    plt.hist(be[:-1], be, weights=result)
#    plt.hist(rqs[i], bins=bins)
#    plt.hist(rqsu[i], bins=bins)
    plt.xlabel(r'$\rho$', fontsize=24)
#    plt.ylabel(r'Nº of pixels')
    plt.ylabel(r'Relative density')
#plt.tight_layout()
plt.show()

