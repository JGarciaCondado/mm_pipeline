import tensorflow as tf

new_model = tf.keras.models.load_model('saved_model/segmentation')
new_model.summary()
