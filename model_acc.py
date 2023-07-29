

from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np

# 생성한 데이터셋 로드
data_path = './testing'  # 손글씨 이미지가 저장된 폴더 경로
labels = []  # 레이블을 저장할 리스트
images = []  # 이미지 데이터를 저장할 리스트

for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.PNG') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경

        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #image[image <= 15] = 0
        image = cv2.resize(image, (28, 28))  # 이미지 크기 조정
        image = image.astype('float32') / 255.0  # 정규화

        images.append(image)
        label = int(filename[-5])
        labels.append(label)

test_images = np.array(images)
test_labels = np.array(labels)

# test 데이터셋 크기
#data_size = len(test_images)
#print(data_size)

test_images = test_images.reshape((-1, 28, 28, 1))
test_images = test_images.astype('float32') / 255.0

# 모델 파일 로드
model = load_model('./model/cnn_model_16.h5')

print(model.summary())

# 입력 데이터에 대한 예측
predictions = model.predict(test_images)

# 예측된 클래스 레이블 가져오기
predicted_labels = predictions.argmax(axis=1)


# 예측 결과 출력
print(predicted_labels)
#정답 출력
print(test_labels)


#틀린 경우 출력
for i in range(100):
    if(predicted_labels[i] != test_labels[i]):
        print('예상', predicted_labels[i])
        print('정답', test_labels[i])
        print('=================')
        #print( predictions[i])

# 정확도 계산
accuracy = (predicted_labels == test_labels).mean()
print("Accuracy:", accuracy)
