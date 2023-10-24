import tensorflow as tf
import numpy as np
import random
import cv2
import albumentations as A
from sklearn.preprocessing import MinMaxScaler
import os

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, data_path, img_size=(224, 224), augmentation=1, shuffle=True):
        # Initialize the CustomDataGen class with data path and other parameters
        self.data_path = data_path
        self.img_size = img_size
        self.shuffle = shuffle
        self.augmentation = augmentation

        # Define image transformation pipeline using Albumentations library
        self.transform = A.Compose([
            A.HorizontalFlip(),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=1, p=0.3),
                A.Blur(blur_limit=1, p=0.3),
            ], p=0.4),
            A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=1, val_shift_limit=1, p=0.3)])


        # Collect image file paths and age labels from the specified data directory
        self.image_paths = []
        self.age_labels = []
        for file_name in os.listdir(data_path):
            if file_name.endswith(".jpg"):
                self.image_paths.append(os.path.join(data_path, file_name))
                self.age_labels.append(self.path_to_age(file_name))

        self.n = len(self.image_paths)

    def preprocess(self, img):
        # Preprocess an image by resizing it and normalizing pixel values
        resized = cv2.resize(img, self.img_size)
        n_resized = resized / 255.0
        return n_resized

    def on_epoch_end(self):
        # Shuffle the image paths and age labels at the end of each epoch
        if self.shuffle:
            temp = list(zip(self.image_paths, self.age_labels))
            random.shuffle(temp)
            self.image_paths, self.age_labels = zip(*temp)
            self.image_paths, self.age_labels = list(self.image_paths), list(self.age_labels)

    def path_to_age(self, path):
        # Extract age information from the file name
        return int(path.split('age')[-1].split('.')[0])

    def __getitem__(self, index):
        X_All = []
        y_All = []
        current_image_path = self.image_paths[index]
        X = cv2.imread(current_image_path)
        y = self.age_labels[index]

        if self.augmentation == 1:
            # If no augmentation, preprocess the image and collect it
            X = self.preprocess(X)
            X_All.append(X)
            y_All.append(np.expand_dims(y, 0))
        else:
            # If augmentation is requested, apply transformations to create multiple variations
            for index in range(self.augmentation):
                X1 = self.transform(image=X)
                new_img = X1["image"]
                X_All.append(self.preprocess(new_img))
                y_All.append(np.expand_dims(y, 0))

        X_All = np.array(X_All)
        y_All = np.array(y_All)

        return X_All, y_All

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.n
