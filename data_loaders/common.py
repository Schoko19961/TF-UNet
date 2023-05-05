import tensorflow as tf
import random
from typing import Tuple

random.seed(1234)

def randomize_image(image, seed: int, apply_color_changes: bool):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.keras.layers.RandomRotation(factor = 0.2,seed=seed)(image)
    if apply_color_changes:
        image = tf.image.random_contrast(image, lower=.4, upper=.6,seed=seed)
        image = tf.image.random_brightness(image, max_delta=.2,seed=seed)
        image = tf.image.random_saturation(image, lower=.4, upper=.6,seed=seed)
        image = tf.image.random_hue(image, max_delta=.2,seed=seed)
    return image

def randomize_images(image, mask):
    seed = random.randint(0, 100000000)
    image = randomize_image(image, seed, True)
    mask = randomize_image(mask, seed, False)
    return image, mask

def resize(image, mask, input_shape: Tuple[int, int]):
    image = tf.image.resize(image,input_shape)
    mask = tf.image.resize(mask,input_shape)
    return image / 255, mask / 255

def load_images(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_image(mask, channels=0, expand_animations = False)
    mask = tf.image.rgb_to_grayscale(mask)
    return image, mask