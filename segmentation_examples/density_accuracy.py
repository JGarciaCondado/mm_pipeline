import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

images = np.load('im_stack_pd.npy')
images_gt = np.load('im_gt_stack_pd.npy')

densities = [1100, 1300, 1500, 1700, 1900, 2100, 2300]
photons = [10, 15, 20, 25, 30, 35, 40, 45, 50]

model = tf.keras.models.load_model('../saved_model/segmentation')
accuracy = np.empty((len(densities), len(photons)))
for i in range(len(densities)):
    for j in range(len(photons)):
        ims = images[:, i, j, :, :]
        ims = np.array([(im - np.min(im)) / (np.max(im) - np.min(im)) for im in ims])
        ims_gt = images_gt[:,i,j,:,:]
        dataset = tf.data.Dataset.from_tensor_slices((ims[...,tf.newaxis], ims_gt[..., tf.newaxis]))
        loss, acc = model.evaluate(dataset.batch(100))
        accuracy[i,j] = acc

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = photons
Y = densities
X, Y = np.meshgrid(X, Y)
Z = accuracy

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
