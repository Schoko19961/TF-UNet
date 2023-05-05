from data_loaders.carvana.data_loader import get_dataset
from unet.model import Unet
import os
import numpy as np 
import tensorflow as tf
from pathlib import Path
from time import strftime
from PIL import Image

IMG_HEIGHT = 960
IMG_WIDTH = 640
IMG_CHANNELS = 3
BATCH_SIZE = 2
NUM_EPOCHS = 200000
TRAIN_IMAGE_DIR = os.path.join(os.getcwd(),"datasets/carvana-image-segmentation/data/train/images")
TRAIN_MASKS_DIR = os.path.join(os.getcwd(),"datasets/carvana-image-segmentation/data/train/masks")
CHECKPOINTS_DIR = os.path.join(os.getcwd(),"checkpoints")
RES_DIR = os.path.join(os.getcwd(),"datasets/carvana-image-segmentation/results")

def get_run_logdir(root_logdir=os.path.join(os.getcwd(),"run_logs")):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")


def train_model(load_last: bool = False):
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=400,
        decay_rate=0.96
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = Unet(img_shape=input_shape,levels=[64, 128, 256, 512])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    if load_last:
        weights = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
        model = model.load_weights(weights)

    train, val = get_dataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASKS_DIR, input_shape = (IMG_WIDTH, IMG_HEIGHT), randomize=True, batch_size=BATCH_SIZE, val_split=.05)

    checkoint_dir = get_run_logdir(CHECKPOINTS_DIR) 
    log_dir = get_run_logdir() 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir,
                                                profile_batch=(100, 200))

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkoint_dir, save_weights_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True)
    
    print("train")
    model.fit(train, epochs=NUM_EPOCHS, validation_data=val, callbacks=[checkpoint_cb,early_stopping_cb, tensorboard_cb])

    return train, val, model

if __name__ == "__main__":
    train_data, val_data, model = train_model(False)
    print("predict")
    results = model.predict(val_data) * 255
    results = results.astype(np.uint8)

    print("Save images")
    RES_MASKS = os.path.join(RES_DIR, "masks")
    for index, res in enumerate(results):
        res = res[:,:,0]
        res = Image.fromarray(res).convert("RGB")
        print(index)
        res.save(os.path.join(RES_MASKS, f"res{index}.jpg"))
    RES_SOURCE = os.path.join(RES_DIR, "source")
    
    index = 0
    for image, mask in val_data.unbatch():
        res = image * 255
        res = res.numpy()
        print(index)
        res = res.astype(np.uint8)
        # res = res[0,:,:,:]
        res = Image.fromarray(res)
        res.save(os.path.join(RES_SOURCE, f"res{index}.jpg"))
        index += 1