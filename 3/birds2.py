from __future__ import division
from matplotlib.image import imread
import keras
import numpy as np
import os
from utils import HistoryCheckpoint
from keras import callbacks, backend
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

backend.clear_session()
print('Using Keras version', keras.__version__)

img_size = 128
n_batch_size = 32
n_epochs = 400
n_base_img = 3000
n_test_img = 3033
n_validation_img = 900
input_shape = (img_size, img_size, 3)

# Two hidden layers
nn = Sequential()

nn.add(Conv2D(64, (3, 3), strides=2, activation='relu', input_shape=input_shape))

nn.add(Conv2D(64, (3, 3), strides=2, activation='relu'))

nn.add(Conv2D(32, (3, 3), activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))

nn.add(Conv2D(32, (3, 3), activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))

nn.add(Flatten())
nn.add(Dropout(0.5))
nn.add(Dense(32, activation='relu'))
nn.add(Dense(200, activation='softmax'))

nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_path = 'models2/model-{epoch:02d}-{val_loss:.2f}.hdf5'
if not os.path.exists('models2'):
    os.makedirs('models2')
history_path = 'history2/model-{epoch}.json'
if not os.path.exists('history2'):
    os.makedirs('history2')
dump_period = 100
save_model_callback = callbacks.ModelCheckpoint(model_path, period=dump_period)
save_history_callback = HistoryCheckpoint(history_path, period=dump_period)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
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

nn.fit_generator(
        train_generator,
        steps_per_epoch=n_base_img//n_batch_size,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=n_validation_img//n_batch_size,
        callbacks=[save_model_callback, save_history_callback])