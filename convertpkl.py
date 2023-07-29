import pickle
import numpy as np
import sys, os
import cv2


labels = []  # 예시: 라벨 정보
images = []

data_path = './realFinal'  # 손글씨 이미지가 저장된 폴더 경로

for filename in os.listdir(data_path):
    if filename.endswith('.png') or filename.endswith('.PNG') or filename.endswith('.jpg'):  # 이미지 파일 확장자에 맞게 변경
        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

        label = int(filename[0])
        labels.append(label)

# 이미지와 라벨을 딕셔너리로 저장
data = {
    'images': np.array(images),
    'labels': np.array(labels)
}

# 피클 파일로 저장
pickle_path = './pkl/FinalTestdata.pkl'
with open(pickle_path, 'wb') as file:
    pickle.dump(data, file)
