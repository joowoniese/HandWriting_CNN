from PIL import Image

# 이미지 열기


from PIL import Image
import numpy as np
import cv2

import sys, os

from tensorflow.python.estimator import keras


data_path = './real'  # 손글씨 이미지가 저장된 폴더 경로

for filename in os.listdir(data_path):
    if filename.endswith('.jpg'):

        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 이미지의 크기 가져오기
        height, width = image.shape[:2]

        # 테두리를 자를 영역 설정 (상단: 10 픽셀, 하단: 10 픽셀, 좌측: 10 픽셀, 우측: 10 픽셀)
        top = 10
        bottom = 10
        left = 10
        right = 10

        # 테두리를 제외한 이미지 잘라내기
        cropped_image = image[top:height - bottom, left:width - right]

        cv2.imwrite('./real' + filename, cropped_image)
