from networks.cellnet import MMNet
from contour import boundary
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import pandas as pd
import matplotlib.pyplot as plt


# Load examples and labels
examples = np.load('dataset/im_stack.npy')[:300]
targets = np.load('dataset/params.npy')[:300]

DATASET_SIZE = examples.shape[0]

# Normalize and add channel dimension
examples = np.array([(example - np.min(example)) / (np.max(example) - np.min(example)) for example in examples])
examples = examples[..., tf.newaxis]
targets = np.array([[r, l, R, theta, centroid[0], centroid[1]] for r,l,R,theta, centroid in targets])

# Create full dataset
dataset = tf.data.Dataset.from_tensor_slices((examples, targets))

# Split into train and test
train_size = int(0.7*DATASET_SIZE)
val_size = int(0.15*DATASET_SIZE)
test_size = int(0.15*DATASET_SIZE)
dataset = dataset.shuffle(DATASET_SIZE)
BATCH_SIZE = 32
train_dataset = dataset.take(train_size).batch(BATCH_SIZE)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.take(val_size).batch(1)
test_dataset = test_dataset.skip(val_size)

model = MMNet().create_model((50,26,1))
model.summary()
optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
EPOCHS = 700
VALIDATION_STEPS = 32

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

early_history = model.fit(train_dataset,
                          epochs=EPOCHS,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=val_dataset,
                          verbose =0,
                          callbacks=[early_stop, tfdocs.modeling.EpochDots()])

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Early Stopping': early_history}, metric = "mae")
plt.ylim([0, 15])
plt.ylabel('MAE [MPG]')
plt.show()

loss, mae, mse = model.evaluate(test_dataset.batch(1),verbose=2)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))


# Plot boundaries
test_predictions = model.predict(test_dataset.batch(1))
test_labels = []
test_images = []
for image, params in test_dataset:
    test_images.append(image)
    test_labels.append(params)
test_labels = np.array(test_labels)

#for i in range(1):
#    r, l, R, theta, centroid1, centroid2 = test_labels[i]
#    plt.imshow(test_images[i][:, :, 0])
#    Boundary = boundary(r,l,R,theta)
#    verts_spline = Boundary.get_spline(40, 4.4, [centroid1, centroid2])
#    verts_boundary = Boundary.get_boundary(40, 4.4, [centroid1, centroid2])
#    plt.plot(verts_spline[:, 0], verts_spline[:, 1])
#    plt.plot(verts_boundary[:, 0], verts_boundary[:, 1])
#    r, l, R, theta, centroid1, centroid2 = test_predictions[i]
#    Boundary = boundary(r,l,R,theta)
#    verts_spline = Boundary.get_spline(40, 4.4, [centroid1, centroid2])
#    verts_boundary = Boundary.get_boundary(40, 4.4, [centroid1, centroid2])
#    plt.plot(verts_spline[:, 0], verts_spline[:, 1])
#    plt.plot(verts_boundary[:, 0], verts_boundary[:, 1])
#    plt.show()

params_label = ['r', 'l', 'R', 'theta', 'x-centroid', 'y-centroid']
lims = [(0.4, 0.6), (1, 6), (-200, 200), (-20, 20), (0, 26), (0,50)]

# Plot true vs prediction
plt.figure(figsize=(15,15))
for i in range(len(params)):
    plt.subplot(2, 3, i+1)
    plt.scatter(test_labels[:, i], test_predictions[:, i])
    plt.xlabel('True Values [{}]'.format(params_label[i]))
    plt.ylabel('Predictions[{}]'.format(params_label[i]))
    plt.xlim(lims[i])
    plt.ylim(lims[i])
    plt.plot(lims, lims, 'r')
plt.show()

# Plot errors
plt.figure(figsize = (15,15))
for i in range(len(params)):
    plt.subplot(3,2,i+1)
    error = test_predictions[:,i] - test_labels[:,i]
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [{}]".format(params_label[i]))
    plt.ylabel("Count")
plt.show()
