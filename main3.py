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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 생성한 데이터셋 로드
data_path = './final_dataset'  # 손글씨 이미지가 저장된 폴더 경로
labels = []  # 레이블을 저장할 리스트
images = []  # 이미지 데이터를 저장할 리스트

for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.PNG') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))  # 이미지 크기 조정

        images.append(image)
        label = int(filename[0])
        labels.append(label)


data_path = './augM'  # 손글씨 이미지가 저장된 폴더 경로

for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.PNG') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))  # 이미지 크기 조정

        images.append(image)
        label = int(filename[0])
        labels.append(label)


data_path = './firstTest'  # 손글씨 이미지가 저장된 폴더 경로

for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.PNG') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))  # 이미지 크기 조정

        images.append(image)
        label = int(filename[-5])
        labels.append(label)

custom_images = np.array(images)
custom_labels = np.array(labels)


hand_path = './real_handwriting'  # 손글씨 이미지가 저장된 폴더 경로
hand_labels = []  # 레이블을 저장할 리스트
hand_images = []  # 이미지 데이터를 저장할 리스트

for filename in os.listdir(hand_path):
    if filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        hand_image_path = os.path.join(hand_path, filename)
        hand_image = cv2.imread(hand_image_path, cv2.IMREAD_GRAYSCALE)
        hand_image = cv2.resize(hand_image, (28, 28))  # 이미지 크기 조정

        hand_images.append(hand_image)
        hand_label = int(filename[-5])
        hand_labels.append(hand_label)

hand_images = np.array(hand_images)
hand_labels = np.array(hand_labels)


# MNIST 데이터셋 로드
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# 데이터셋 통합
combined_images = np.concatenate((custom_images, train_images, test_images, hand_images), axis=0)
combined_labels = np.concatenate((custom_labels, train_labels, test_labels, hand_labels), axis=0)

# 데이터 전처리
combined_images = combined_images.astype('float32') / 255.0


# 데이터셋 크기
data_size = len(combined_images)
print(data_size)

# 랜덤하게 데이터 인덱스를 섞음
shuffled_indices = np.random.permutation(data_size)

# 검증 데이터셋 크기
# 전체 데이터셋 크기의 15%
test_size = int(0.15 * data_size)

# xtest dataset
test_indices = shuffled_indices[:test_size]
final_test_images = combined_images[test_indices]
final_test_labels = combined_labels[test_indices]

# 학습 데이터셋 추출
train_indices = shuffled_indices[test_size:]
final_train_images = combined_images[train_indices]
final_train_labels = combined_labels[train_indices]

# 모델 구성
model = Sequential()
model.add(Conv2D(20, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(AveragePooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(AveragePooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

hist = model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(final_train_images, final_train_labels, epochs=60, batch_size=30, validation_split=0.15)

#loss_hist = []
#acc_hist = []

#loss_hist.append(hist.history['loss'])
#acc_hist.append(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.show()

plt.plot(hist.history['accuracy'])
plt.show()

# 먼저 테스트셋으로 정확도 검사
loss, accuracy = model.evaluate(final_test_images, final_test_labels, verbose=1)
print('Test Accuracy:', accuracy)

# 모델 저장
model.save('./model/cnn_model_58.h5')