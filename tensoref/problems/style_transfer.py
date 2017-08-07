import numpy as np
import tensorflow as tf

from tensoref.config import CONFIG


def run(content, style, image_width, image_height):
    input_image = tf.Variable(np.zeros([1, image_width, image_height, 3]), dtype=tf.float32)
