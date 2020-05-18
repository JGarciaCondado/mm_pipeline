import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import rc
rc('text', usetex=True)
import sys
sys.path.append('../')
from molyso.generic.otsu import threshold_otsu

cells = np.load('../dataset/im_stack.npy')[:1000]
images_gt = np.load('../dataset/im_gt_stack.npy')[:1000]

model = tf.keras.models.load_model('../saved_model/segmentation')
accuracy_ml = []
acc_otsu = []
for im, im_gt in zip(cells, images_gt):
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        dataset = tf.data.Dataset.from_tensor_slices((im[tf.newaxis, ...,tf.newaxis], im_gt[tf.newaxis, ..., tf.newaxis]))
        loss, acc = model.evaluate(dataset.batch(1))
        accuracy_ml.append(acc)
        binary_image = im > threshold_otsu(im)
        acc_otsu.append(1 - np.sum(binary_image != im_gt)/im.size)

print(np.mean(accuracy_ml))
print(np.min(accuracy_ml))
print(np.max(accuracy_ml))
print(np.mean(acc_otsu))
print(np.min(acc_otsu))
print(np.max(acc_otsu))
