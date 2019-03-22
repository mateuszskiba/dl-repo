from __future__ import division
from matplotlib.image import imread
import keras
import numpy as np
import os
from utils import HistoryCheckpoint
from keras import callbacks, backend
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

backend.clear_session()
print('Using Keras version', keras.__version__)

img_size = 128
n_batch_size = 32
n_epochs = 250
n_base_img = 3000
n_test_img = 3033
n_validation_img = 900
input_shape = (img_size, img_size, 3)
old_model_path = 'models2/model-150-7.05.hdf5'

model_path = 'models-re2/model-{epoch:02d}-{val_loss:.2f}.hdf5'
if not os.path.exists('models-re2'):
    os.makedirs('models-re2')
history_path = 'history-re2/model-{epoch}.json'
if not os.path.exists('history-re2'):
    os.makedirs('history-re2')
dump_period = 25
save_model_callback = callbacks.ModelCheckpoint(model_path, period=dump_period)
save_history_callback = HistoryCheckpoint(history_path, period=dump_period)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0)

train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(img_size, img_size),
        batch_size=n_batch_size,
        class_mode='categorical')

test_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='constant',
        cval=0)

validation_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(img_size, img_size),
        batch_size=n_batch_size,
        class_mode='categorical')

nn = load_model(old_model_path)

nn.fit_generator(
        train_generator,
        steps_per_epoch=n_base_img//n_batch_size,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=n_validation_img//n_batch_size,
        callbacks=[save_model_callback, save_history_callback])