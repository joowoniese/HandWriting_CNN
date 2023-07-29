from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
import sys, os

from tensorflow.python.estimator import keras


data_path = './realFinal01'  # 손글씨 이미지가 저장된 폴더 경로
test_labels = []  # 레이블을 저장할 리스트
test_images = []  # 이미지 데이터를 저장할 리스트

for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.PNG') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        image_path = os.path.join(data_path, filename)
        #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        #image = cv2.resize(image, (28, 28))  # 이미지 크기 조정
        image.save('./realFinal/' + filename[-5] + filename)