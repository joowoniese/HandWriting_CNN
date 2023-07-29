from tensorflow.keras.datasets import mnist
from PIL import Image
import os
import random

# MNIST 데이터셋 로드
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Augmentation 수행할 폴더 경로
save_path = './augM'

# Augmentation 수행 함수
def augmentation(image):
    # 이미지 회전
    angle = random.randint(-15, 15)
    rotated_image = image.rotate(angle)

    # 이미지 이동 변환
    trans_x = random.randint(-2, 2)
    trans_y = random.randint(-2, 2)
    transformed_image = rotated_image.transform(image.size, Image.AFFINE, (1, 0, trans_x, 0, 1, trans_y))

    return transformed_image

# MNIST 데이터셋 이미지에 대해 augmentation 수행
for i in range(len(train_images)):
    image = Image.fromarray(train_images[i])
    label = train_labels[i]

    # 원본 이미지 저장
    original_filename = f"{label}_{i}.png"
    original_filepath = os.path.join(save_path, original_filename)

    # augmentation 수행 및 이미지 저장
    for j in range(2):  # augmentation을 2번 수행
        augmented_image = augmentation(image)

        augmented_filename = f"{label}_{i}_aug{j+1}.png"
        augmented_filepath = os.path.join(save_path, augmented_filename)
        augmented_image.save(augmented_filepath)
for i in range(len(test_images)):
    image = Image.fromarray(test_images[i])
    label = test_labels[i]

    # 원본 이미지 저장
    original_filename = f"{label}_{i}.png"
    original_filepath = os.path.join(save_path, original_filename)

    # augmentation 수행 및 이미지 저장
    for j in range(2):  # augmentation을 2번 수행
        augmented_image = augmentation(image)

        augmented_filename = f"{label}_{i}_taug{j+1}.png"
        augmented_filepath = os.path.join(save_path, augmented_filename)
        augmented_image.save(augmented_filepath)