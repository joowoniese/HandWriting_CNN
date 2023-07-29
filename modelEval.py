from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import sys, os

from tensorflow.python.estimator import keras

# 모델 불러오기
model = load_model('./model/cnn_model_49.h5')

data_path = './realFinal'  # 손글씨 이미지가 저장된 폴더 경로
test_labels = []  # 레이블을 저장할 리스트
test_images = []  # 이미지 데이터를 저장할 리스트
cnt = 0

for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.PNG') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))  # 이미지 크기 조정
        image = image.astype('float32') / 255.0  # 정규화

        # input_image = np.expand_dims(normalized_image, axis=0)  # 배치 차원 추가
        test_images.append(image)
        label = int(filename[-5])
        test_labels.append(label)
        
        print("=====================================================================")

        predictions = model.predict(image)
        print(image)
        
        # 예측 결과 출력
        print(predictions)
        predicted_classes = np.argmax(predictions)

        print("예측 : ", predicted_classes)
        print("정답 : ", label)

        if predicted_classes != label:
            cnt += 1

        predicted_probability = predictions[0][predicted_classes]
        predicted_probability_with_e = "{:.16f}".format(predicted_probability)
        print("확률 : ", predicted_probability_with_e)


test_images = np.array(test_images)
test_labels = np.array(test_labels)

'''


data_size = len(test_images)
print(data_size)

# 랜덤하게 데이터 인덱스를 섞음
shuffled_indices = np.random.permutation(data_size)



# xtest dataset
test_indices = shuffled_indices[:200]
test_images = test_images[test_indices]
test_labels = test_labels[test_indices]

'''

loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
print('Test Accuracy :', accuracy)
print('틀린 개수 : ', cnt)