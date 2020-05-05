import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load examples and labels
examples = np.load('dataset/im_stack.npy')[:1000]
labels = np.load('dataset/im_gt_stack.npy')[:1000]

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
BATCH_SIZE = 32
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

class UNet():
    def __init__(self):
        print ('build UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, img_shape, num_class):

        concat_axis = 3
        inputs = layers.Input(shape = img_shape)

        conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

        up_conv3 = layers.UpSampling2D(size=(2, 2))(conv3)
        ch, cw = self.get_crop_shape(conv2, up_conv3)
        crop_conv2 = layers.Cropping2D(cropping=(ch,cw))(conv2)
        up4 = layers.concatenate([up_conv3, crop_conv2], axis=concat_axis)
        conv4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up4)
        conv4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)

        up_conv4 = layers.UpSampling2D(size=(2, 2))(conv4)
        ch, cw = self.get_crop_shape(conv1, up_conv4)
        crop_conv1 = layers.Cropping2D(cropping=(ch,cw))(conv1)
        up5 = layers.concatenate([up_conv4, crop_conv1], axis=concat_axis)
        conv5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(up5)
        conv5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv5)

        ch, cw = self.get_crop_shape(inputs, conv5)
        conv5 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv5)
        conv6 = layers.Conv2D(num_class, (2, 2), activation = 'relu', padding='same')(conv5)
        conv6 = layers.Conv2D(num_class, (1, 1))(conv6)

        model = models.Model(inputs=inputs, outputs=conv6)

        return model

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
model.save('saved_model/segmentation')

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
