import tensorflow as tf
import numpy as np
import cv2
import os
import datetime;
from tensorflow import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from CustomDataGen import CustomDataGen



# ct stores current time
ct = datetime.datetime.now()

# ts store timestamp of current time
ts = ct.timestamp()


# Paths to training and testing data
data_path = "dataset\\train"
test_data_path = "dataset\\test"

# Data loading and splitting
data_generator = CustomDataGen(data_path, img_size=(224, 224), augmentation=1, shuffle=True)
val_data_generator = CustomDataGen(test_data_path, img_size=(224, 224), augmentation=1, shuffle=False)

# Callbacks for model training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='models\\' + str(ts) + '.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs\\' + str(ts), histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None)

# Loading a pre-trained model and building a new model
model = tf.keras.applications.resnet.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    classes=1,
)

x = tf.keras.layers.Flatten()(model.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
out = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(model.input, out)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000001), loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError()])

# Model training
epochs = 10
model.fit(data_generator, validation_data=val_data_generator, epochs=epochs, callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback])
