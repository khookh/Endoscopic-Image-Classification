import tensorflow as tf
import numpy as np


class Random90Rotation(tf.keras.layers.Layer):
    def __init__(self, movements=[0, 1, 2, 3], **kwargs):
        super(Random90Rotation, self).__init__(**kwargs)
        self.movements = movements

    def call(self, images, training=None):
        if not training:
            return images
        images = tf.image.rot90(images, k=np.random.choice(self.movements))
        return images
