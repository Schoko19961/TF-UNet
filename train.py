import albumentations as A
import tqdm
from unet.model import Unet
from scripts.data_loader import get_dataset
import os
from PIL import Image
import numpy as np 
import tensorflow as tf


IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 16
NUM_EPOCHS = 2
TRAIN_IMAGE_DIR = os.path.join(os.getcwd(),"data/train/images")
TRAIN_MASKS_DIR = os.path.join(os.getcwd(),"data/train/masks")
RES_DIR = os.path.join(os.getcwd(),"data/results")

def main():
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    optimizer = tf.keras.optimizers.Adam()
    model = Unet(img_shape=input_shape,levels=[16, 32, 64, 128])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    train = get_dataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASKS_DIR, input_shape = (IMG_WIDTH, IMG_HEIGHT), randomize_images=True).take(400)
    val = get_dataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASKS_DIR, input_shape = (IMG_WIDTH, IMG_HEIGHT), randomize_images=False).skip(400).take(10)

    
    print("train")
    model.fit(train, epochs=NUM_EPOCHS, shuffle=False)
    print("predict")

    results = model.predict(val)* 255
    results = results.astype(np.uint8)
    print("Save images")
    for index, res in enumerate(results):
        res = res[:,:,0]
        res = Image.fromarray(res).convert("RGB")
        res.save(os.path.join(RES_DIR, f"res{index}.jpg"))

if __name__ == "__main__":
    main()