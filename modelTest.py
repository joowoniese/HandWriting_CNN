import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import sys, os

pickle_path = './pkl/testdata2D.pkl'

with open(pickle_path, 'rb') as file:
    data = pickle.load(file)

images = data[0]
labels = data[1]

model = load_model('./model/cnn_model_48.h5')
test_images = []
test_labels = []

cnt = 0
cnt1 = 0
cnt2 = 0


for i in range(len(images)):
    image = images[i]
    #print(image)
    label = labels[i]

    image = cv2.resize(image, (28, 28))  # 이미지 크기 조정
    image = image.astype('float32') / 255.0  # 정규화

    test_images.append(image)
    test_labels.append(label)


    print("=====================================================================")
    image = image.reshape((-1, 28, 28, 1))
    image = image.astype('float32') / 255.0

    predictions = model.predict(image)
    #print(image)

    predicted_label = np.argmax(predictions)
    print('Input Label:', label)
    print('Predicted Label:', predicted_label)

    label = int(label)
    predicted_label = int(predicted_label)


    if predicted_label != label:
        cnt += 1
        if label == 1:
            cnt1 += 1
        if label == 2:
            cnt2 += 1


test_images = np.array(test_images)
test_labels = np.array(test_labels)

test_images = test_images.reshape((-1, 28, 28, 1))
test_images = test_images.astype('float32') / 255.0

loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
print('Test Accuracy :', accuracy)
print('wrong : ', cnt)
print('one : ', cnt1)
print('two : ', cnt2)
