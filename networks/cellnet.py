import tensorflow as tf
from tensorflow.keras import layers, models
import sys
sys.path.append('../')
from contour import bacteria_polygon


class CellNet():
    def __init__(self, m_ratio):
        self.m_ratio = m_ratio
        print('build CellNet...')

    def loss(self, target_params, predicted_params):
       # target_params, predicted_params = target_params.numpy(), predicted_params.numpy()
        total_loss = 0.0
        for i in range(len(target_params)):
            target_boundary = bacteria_polygon(target_params[i], self.m_ratio)
            predicted_boundary = bacteria_polygon(predicted_params[i], self.m_ratio)
            loss = target_boundary.diff_area(predicted_boundary.polygon)
            total_loss += loss
        return total_loss

    def create_model(self):
        inputs = layers.Input(shape = (50,26,1))
        conv1 = layers.Conv2D(6, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        flatten = layers.Flatten()(pool2)
        fc1 = layers.Dense(128, activation='relu')(flatten)
        fc2 = layers.Dense(64, activation='relu')(fc1)
        fc3 = layers.Dense(6)(fc2)

        model = models.Model(inputs=inputs, outputs=fc3)

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

class MMNet():
    def __init__(self):
        print('build MMNet...')

    def create_model(self, img_shape):
        inputs = tf.keras.layers.Input(shape = img_shape)
        down_stack = tf.keras.models.load_model('saved_model/segmentation')
        down_stack.trainable = False
        x = inputs
        model_op = tf.argmax(down_stack(x), axis=-1)
        flatten = layers.Flatten()(model_op)
        fc1 = layers.Dense(128, activation='relu')(flatten)
        fc2 = layers.Dense(64, activation='relu')(fc1)
        fc3 = layers.Dense(6)(fc2)

        return tf.keras.Model(inputs=inputs, outputs=fc3)
