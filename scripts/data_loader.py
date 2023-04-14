from typing import Tuple
import os
import numpy as np
import tensorflow as tf
import random
from PIL import Image

random.seed(1234)
tf.random.set_seed(1234)


def augment_image(image, seed: int, apply_color_changes: bool):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.keras.layers.RandomRotation(factor = 0.2,seed=seed)(image)
    if apply_color_changes:
        image = tf.image.random_contrast(image, lower=.4, upper=.6,seed=seed)
        image = tf.image.random_brightness(image, max_delta=.2,seed=seed)
        image = tf.image.random_saturation(image, lower=.4, upper=.6,seed=seed)
        image = tf.image.random_hue(image, max_delta=.2,seed=seed)
    return image

def randomize(image, mask):
    seed = random.randint(0, 100000000)
    image = augment_image(image, seed, True)
    mask = augment_image(mask, seed, False)
    return image, mask

def resize(image, mask, input_shape: Tuple[int, int]):
    print(input_shape)
    image = tf.image.resize(image,input_shape)
    # image = tf.image.per_image_standardization(image)
    mask = tf.image.resize(mask,input_shape)
    # mask = tf.image.per_image_standardization(mask)
    return image / 255, mask / 255

def load_images(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_image(mask, channels=0, expand_animations = False)
    mask = tf.image.rgb_to_grayscale(mask)
    return image, mask

RES_DIR = os.path.join(os.getcwd(),"data/results")


def get_dataset(image_dir, mask_dir, input_shape: Tuple[int, int], randomize_images=True) -> tf.data.Dataset:
    images = os.listdir(image_dir)
    masks = [image.replace(".jpg", "_mask.gif") for image in images]
    images = [os.path.join(image_dir,image) for image in images]
    masks = [os.path.join(mask_dir,mask) for mask in masks]

    dataset = tf.data.Dataset.from_tensor_slices((images, masks)).map(map_func=load_images, num_parallel_calls=tf.data.AUTOTUNE)
    if randomize_images:
        dataset = dataset.map(map_func=randomize, num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.map(map_func=lambda image, mask: resize(image, mask, input_shape), num_parallel_calls=tf.data.AUTOTUNE).batch(10).prefetch(1)

    # for index, res in enumerate(dataset.take(3)):
    #     res = res[0][0].numpy()
    #     res = res * 255
    #     res = res.astype(np.uint8)
    #     res = Image.fromarray(res).convert("RGB")
    #     res.save(os.path.join(RES_DIR, f"res{index}.jpg"))
    return dataset
