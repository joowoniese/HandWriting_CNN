import random
import numpy as np
from keras.layers import Dropout, AveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import cv2
import matplotlib.pyplot as plt


data_path = './final_dataset'
labels = []
images = []

for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.PNG') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))

        images.append(image)
        label = int(filename[0])
        labels.append(label)


hand_path = './real_handwriting'

for filename in os.listdir(hand_path):
    if filename.endswith('.jpg'):

        hand_image_path = os.path.join(hand_path, filename)
        hand_image = cv2.imread(hand_image_path, cv2.IMREAD_GRAYSCALE)
        hand_image = cv2.resize(hand_image, (28, 28))

        images.append(hand_image)
        hand_label = int(filename[-5])
        labels.append(hand_label)

custom_images = np.array(images)
custom_labels = np.array(labels)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

combined_images = np.concatenate((custom_images, train_images, test_images), axis=0)
combined_labels = np.concatenate((custom_labels, train_labels, test_labels), axis=0)

combined_images = combined_images.astype('float32') / 255.0

data_size = len(combined_images)
print(data_size)

shuffled_indices = np.random.permutation(data_size)

test_size = int(0.15 * data_size)

test_indices = shuffled_indices[:test_size]
final_test_images = combined_images[test_indices]
final_test_labels = combined_labels[test_indices]

train_indices = shuffled_indices[test_size:]
final_train_images = combined_images[train_indices]
final_train_labels = combined_labels[train_indices]

model = Sequential()
model.add(Conv2D(24, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(AveragePooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(AveragePooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

hist = model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(final_train_images, final_train_labels, epochs=60, batch_size=30, validation_split=0.15)

plt.plot(hist.history['loss'])
plt.show()

plt.plot(hist.history['accuracy'])
plt.show()

loss, accuracy = model.evaluate(final_test_images, final_test_labels, verbose=1)
print('Test Accuracy:', accuracy)

model.save('./model/cnn_model_57.h5')