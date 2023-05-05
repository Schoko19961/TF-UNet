from typing import Tuple
import os
import tensorflow as tf
import random
import math
from data_loaders.common import load_images, randomize_images, resize


def get_dataset(image_dir, mask_dir, input_shape: Tuple[int, int], randomize=True, batch_size: int = 16, val_split: float = .05) -> tf.data.Dataset:
    items = os.listdir(image_dir)
    images = [os.path.join(image_dir,image) for image in items]
    masks = [os.path.join(mask_dir,mask) for mask in items]
    data = list(zip(images,masks))
    random.shuffle(data)
    
    train_size = math.floor(len(images) * (1-val_split))
    train_data = data[:train_size]
    # repeat train_data three times for more variations
    train_data = train_data * 3
    random.shuffle(train_data)

    val_data = data[train_size:]
    train_images, train_masks = zip(*train_data)
    val_images, val_masks = zip(*val_data)

    train_dataset = tf.data.Dataset.from_tensor_slices((list(train_images), list(train_masks))).map(map_func=load_images, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((list(val_images), list(val_masks))).map(map_func=load_images, num_parallel_calls=tf.data.AUTOTUNE)
    if randomize:
        train_dataset = train_dataset.map(map_func=lambda image, mask: randomize_images(image, mask, True), num_parallel_calls=tf.data.AUTOTUNE)
    
    train_dataset = train_dataset.map(map_func=lambda image, mask: resize(image, mask, input_shape), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(1)
    val_dataset = val_dataset.map(map_func=lambda image, mask: resize(image, mask, input_shape), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(1)


    return train_dataset, val_dataset
