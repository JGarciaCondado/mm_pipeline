import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
target_mean, target_std  = np.mean(targets, axis=0), np.std(targets, axis=0)
targets = np.array([(target - target_mean)/target_std for target in targets])

# Create full dataset
dataset = tf.data.Dataset.from_tensor_slices((examples, targets))

# Split into train and test
train_size = int(0.7*DATASET_SIZE)
val_size = int(0.15*DATASET_SIZE)
test_size = int(0.15*DATASET_SIZE)
dataset = dataset.shuffle(DATASET_SIZE)
BATCH_SIZE = 1
train_dataset = dataset.take(train_size).batch(BATCH_SIZE)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.take(val_size).batch(1)
test_dataset = test_dataset.skip(val_size)

def build_model():
  model = keras.Sequential([
    layers.Flatten(input_shape=(50, 26, 1)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(6)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()
example_batch = test_dataset.take(10)
example_result = model.predict(example_batch.batch(1))
print(example_result)

EPOCHS = 1000
VALIDATION_STEPS = 32

history = model.fit(
  train_dataset,
  epochs=EPOCHS,
  validation_steps=VALIDATION_STEPS,
  validation_data=val_dataset,
  verbose =0,
  callbacks=[tfdocs.modeling.EpochDots()])

example_result = model.predict(example_batch.batch(1))

print(example_result)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 1])
plt.ylabel('MAE [MPG]')
plt.show()
plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 1])
plt.ylabel('MSE [MPG^2]')
plt.show()
