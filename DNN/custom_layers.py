import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class Random90Rotation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Random90Rotation, self).__init__(**kwargs)

    def call(self, images, training=None):
        if not training:
            return images
        images = tf.image.rot90(images, k=np.random.choice([0, 1, 2, 3]))
        return images


class RandomHue(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomHue, self).__init__(**kwargs)

    def call(self, images, training=None):
        if not training:
            return images
        a = -1.5 / 180  # hue levels are between 0 and 180, standard deviation observed is about 3
        b = -a
        images = tf.image.adjust_hue(images, ((b - a) * np.random.random_sample() + a))
        return images


class RandomGaussian(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomGaussian, self).__init__(**kwargs)

    # def call(self, images, training=None):
    #    if not training:
    #        return images
    #    prob = abs(np.random.normal(0,1))
    #    if 0.4 < prob < 0.8:
    #        images = tf.gaussian.GaussianBlur(size=3)
    #    elif prob >= 0.8:
    #        images = tf.gaussian.GaussianBlur(size=5)
    #    return images
    def call(self, images, training=None):
        if not training:
            return images
        return tfa.image.gaussian_filter2d(image = images, filter_shape=(3,3))
