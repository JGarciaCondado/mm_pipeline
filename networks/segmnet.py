from tensorflow.keras import layers, models
import tensorflow as tf

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

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model
