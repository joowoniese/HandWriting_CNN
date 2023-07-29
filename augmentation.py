from PIL import ImageEnhance
import numpy as np
import sys, os
import cv2
import random
from PIL import Image
from tensorflow.keras.datasets import mnist

data_path = './firstTest'  # 손글씨 이미지가 저장된 폴더 경로


for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        image_path = os.path.join(data_path, filename)
        image = Image.open(image_path).convert('L')

        image1 = image.rotate(15)
        image2 = image.rotate(-15)

        trans_1 = random.randint(-2, 2)
        trans_2 = random.randint(-2, 2)

        image1 = image1.transform(image.size, Image.AFFINE, (1, 0, trans_1, 0, 1, 0))
        image2 = image2.transform(image.size, Image.AFFINE, (1, 0, trans_2, 0, 1, 0))

        #image1 = image.resize((new_width, new_height), Image.ANTIALIAS)
        #image2 = image.resize((new_width, new_height), Image.ANTIALIAS)


        #resizing(전체에서 50-60)

        image1.save("./final_dataset/" + filename[-5] + "aug1" + filename)
        image2.save("./final_dataset/" + filename[-5] + "aug2" + filename)

