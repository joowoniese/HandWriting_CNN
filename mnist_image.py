from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

import sys, os

def save_mnist_image(image_array, save_path):
    # 이미지 배열을 0-255 범위로 변환
    image_array = (image_array * 255).astype(np.uint8)

    # 이미지 객체 생성
    image = Image.fromarray(image_array)

    # 이미지 저장
    image.save(save_path)




pickle_path = './pkl/testdata2D.pkl'

with open(pickle_path, 'rb') as file:
    data = pickle.load(file)

images = data[0]
labels = data[1]

for i in range(len(images)):
    image = images[i]
    image = cv2.resize(image, (28, 28))  # 이미지 크기 조정
    image = image.astype('float32') / 255.0  # 정규화
    save_mnist_image(image, "./img/" + str(i) + ".png")
