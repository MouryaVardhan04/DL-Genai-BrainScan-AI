import os
import numpy as np
import random
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import ssl

# Disable SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Base directory for dataset
base_dir = 'Datasets/MRI Images'
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Testing')

print(f"Training directory: {train_dir}")
print(f"Testing directory: {test_dir}")

# Function to count images per class in a directory
def count_images_per_class(directory):
    from collections import defaultdict
    class_counts = defaultdict(int)
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            class_counts[label] = len([
                f for f in os.listdir(label_dir)
                if os.path.isfile(os.path.join(label_dir, f))
            ])
    return class_counts

# Get image counts
train_counts = count_images_per_class(train_dir)
test_counts = count_images_per_class(test_dir)

print('Training set:')
for label, count in train_counts.items():
    print(f'{label}: {count} images')

print('\nTesting set:')
for label, count in test_counts.items():
    print(f'{label}: {count} images')

# Load and shuffle the train data
train_paths = []
train_labels = []
for label in os.listdir(train_dir):
    for image in os.listdir(os.path.join(train_dir, label)):
        train_paths.append(os.path.join(train_dir, label, image))
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

# Image Augmentation function
def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    image = np.array(image) / 255.0
    return image

# Load images and apply augmentation
def open_images(paths):
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = augment_image(image)
        images.append(image)
    return np.array(images)

# Encoding labels
def encode_label(labels):
    unique_labels = os.listdir(train_dir)
    encoded = [unique_labels.index(label) for label in labels]
    return np.array(encoded)

# Data generator for batching
def datagen(paths, labels, batch_size=12, epochs=1):
    for _ in range(epochs):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = labels[i:i + batch_size]
            batch_labels = encode_label(batch_labels)
            yield batch_images, batch_labels

# Model architecture
IMAGE_SIZE = 128

print("Loading VGG16 base model...")
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Freeze all layers of the VGG16 base model
for layer in base_model.layers:
    layer.trainable = False

# Set the last few layers to be trainable
base_model.layers[-7].trainable = True
base_model.layers[-6].trainable = True
base_model.layers[-5].trainable = True
base_model.layers[-4].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-2].trainable = True

# Build the final model
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(os.listdir(train_dir)), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

print("Model compiled successfully!")

# Parameters
batch_size = 20
steps = int(len(train_paths) / batch_size)
epochs = 5

print(f"Training for {epochs} epochs with {steps} steps per epoch...")

# Train the model
history = model.fit(datagen(train_paths, train_labels, batch_size=batch_size, epochs=epochs),
                    epochs=epochs, steps_per_epoch=steps)

# Save the model
model_save_path = "models/brain_tumor_model.h5"
model.save(model_save_path)
print(f"Model saved successfully to: {model_save_path}")

# Also save the class labels
import pickle
class_labels = ['pituitary', 'notumor', 'glioma', 'meningioma']
with open("models/class_labels.pkl", "wb") as f:
    pickle.dump(class_labels, f)
print("Class labels saved to: models/class_labels.pkl")

print("Training completed successfully!") 