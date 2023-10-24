import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from CustomDataGen import CustomDataGen


# Data path
data_path = "dataset\\train"
test_data_path = "dataset\\test"

# Data loading and splitting
data_generator = CustomDataGen(data_path, img_size=(224, 224), augmentation=1, shuffle=True)
val_data_generator = CustomDataGen(test_data_path, img_size=(224, 224), augmentation=1, shuffle=False)

#Callbacks for model training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints\model_checkpoint.h5', monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs\\', histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None)

# Loading a pre-trained model and building a new model
model = tf.keras.models.load_model('checkpoints\model_checkpoint.h5')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000001), loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError(), 'mse'])

# Model training
epochs = 20
model.fit(data_generator, validation_data=val_data_generator, epochs=epochs,initial_epoch=15, callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback])
