import numpy as np
import tensorflow as tf
from networks.segmnet import UNet
import matplotlib.pyplot as plt

# Load examples and labels
examples = np.load('dataset/im_stack.npy')[:500]
labels = np.load('dataset/im_gt_stack.npy')[:500]

DATASET_SIZE = examples.shape[0]

# Normalize and add channel dimension
examples = np.array([(example - np.min(example)) / (np.max(example) - np.min(example)) for example in examples])
examples = examples[..., tf.newaxis]
labels = labels[..., tf.newaxis]

# Create full dataset
dataset = tf.data.Dataset.from_tensor_slices((examples, labels))

# Split into train and test
train_size = int(0.7*DATASET_SIZE)
val_size = int(0.15*DATASET_SIZE)
test_size = int(0.15*DATASET_SIZE)
dataset = dataset.shuffle(DATASET_SIZE)
BATCH_SIZE = 64
train_dataset = dataset.take(train_size).batch(BATCH_SIZE)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.take(val_size).batch(1)
test_dataset = test_dataset.skip(val_size)

#Model fit

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i][:,:,0])
    plt.axis('off')
  plt.show()

for image, mask in test_dataset.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

model = UNet().create_model((50,26, 1), 2)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image[tf.newaxis, ...])
      display([image, mask, create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
#    clear_output(wait=True)
#    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20
VALIDATION_STEPS = 32
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=val_dataset,
                          callbacks=[DisplayCallback()])
# Save model
#model.save('saved_model/segmentation')

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_dataset, 3)
