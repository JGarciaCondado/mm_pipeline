import tensorflow as tf
from tensorflow.keras import layers, models


class CellNet():
    def __init__(self):
        print('build CellNet...')

    def create_model():
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
