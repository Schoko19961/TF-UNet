from unet.model import Unet
from scripts.data_loader import get_dataset
import os
import numpy as np 
import tensorflow as tf
from pathlib import Path
from time import strftime


IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3
BATCH_SIZE = 16
NUM_EPOCHS = 200000
TRAIN_IMAGE_DIR = os.path.join(os.getcwd(),"data/train/images")
TRAIN_MASKS_DIR = os.path.join(os.getcwd(),"data/train/masks")
CHECKPOINTS_DIR = os.path.join(os.getcwd(),"checkpoints")
RES_DIR = os.path.join(os.getcwd(),"data/results")

def get_run_logdir(root_logdir=os.path.join(os.getcwd(),"run_logs")):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")


def main(load_last: bool = False):
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=250,
        decay_rate=0.96
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = Unet(img_shape=input_shape,levels=[16, 32, 64, 128, 256])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    if load_last:
        weights = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
        model = model.load_weights(weights)

    train = get_dataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASKS_DIR, input_shape = (IMG_WIDTH, IMG_HEIGHT), randomize_images=True).take(2500)
    val = get_dataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASKS_DIR, input_shape = (IMG_WIDTH, IMG_HEIGHT), randomize_images=False).skip(2500)

    checkoint_dir = get_run_logdir(CHECKPOINTS_DIR) 
    log_dir = get_run_logdir() 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir,
                                                profile_batch=(100, 200))

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkoint_dir, save_weights_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights=True)
    
    print("train")
    model.fit(train, epochs=NUM_EPOCHS, validation_data=val, callbacks=[checkpoint_cb,early_stopping_cb, tensorboard_cb])
    print("predict")

    results = model.predict(val)* 255
    results = results.astype(np.uint8)
    print("Save images")
    # for index, res in enumerate(results):
    #     res = res[:,:,0]
    #     res = Image.fromarray(res).convert("RGB")
    #     res.save(os.path.join(RES_DIR, f"res{index}.jpg"))

if __name__ == "__main__":
    main(False)