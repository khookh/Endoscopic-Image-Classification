import tensorflow as tf
import numpy as np


class Random90Rotation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Random90Rotation, self).__init__(**kwargs)

    def call(self, images, training=None):
        if not training:
            return images
        images = tf.image.rot90(images, k=np.random.choice([0, 1, 2, 3]))
        return images
