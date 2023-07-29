import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import sys, os

pickle_path = './pkl/DataSet1_2DN.pkl'

with open(pickle_path, 'rb') as file:
    data = pickle.load(file)

test_images = data[0]
test_labels = data[1]

model = load_model('./model/cnn_model_55.h5')
cnt = 0

test_images = np.array(test_images)
test_labels = np.array(test_labels)

#test_images = test_images.astype('float32') / 255.0
# 입력 데이터에 대한 예측
predictions = model.predict(test_images)

# 예측된 클래스 레이블 가져오기
predicted_labels = predictions.argmax(axis=1)
cnt1 = 0


#틀린 경우 출력
for i in range(len(test_images)):
    print('Input Label', test_labels[i])
    print('Predicted Label', predicted_labels[i])
    print('=================')
    if predicted_labels[i] != test_labels[i]:
        cnt +=1
        image_array = (test_images[i] * 255).astype(np.uint8)
        image = Image.fromarray(image_array)
        image.save("./wrong/" + str(i) + ".png")
        if test_labels[i] == 1:
            cnt1 += 1


#evaluate이용한 정확도
loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
print('Test Accuracy :', accuracy)

# predict 이용한 정확도 계산
accuracy = (predicted_labels == test_labels).mean()
print("Accuracy : ", accuracy)
print("loss : ", loss)
print('틀린 개수 : ', cnt)
print('one : ', cnt1)
print(len(test_images))