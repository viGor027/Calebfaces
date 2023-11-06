import albumentations as A
import cv2
import numpy as np
import os

d = os.path.dirname(os.path.dirname(__file__))
TRAIN_SET_PATH = os.path.join(d, 'data', 'train')
VALID_SET_PATH = os.path.join(d, 'data', 'valid')
TEST_SET_PATH = os.path.join(d, 'data', 'test')
CSV_PATH = os.path.join(d, 'data', 'csv')

CSV_DICT = {
    TRAIN_SET_PATH: os.path.join(d, 'data', 'csv', 'train_cat.csv'),
    VALID_SET_PATH: os.path.join(d, 'data', 'csv', 'valid_cat.csv'),
    TEST_SET_PATH: os.path.join(d, 'data', 'csv', 'test_cat.csv')
}

transform_1 = A.Compose([
    A.RandomBrightnessContrast(p=0.5)
])

transform_2 = A.Compose([
    A.RGBShift(p=0.5)
])

transform_3 = A.Compose([
    A.HueSaturationValue(p=0.5)
])

# img = cv2.imread('../data/test/182637.jpg')
# cv2.imshow('', img)
# cv2.waitKey(0)
# img = transform_3(image=img)['image']
# cv2.imshow('', img)
# cv2.waitKey(0)

TRANSFORMATIONS = [transform_1, transform_2, transform_3]

CLASSES = [0, 1]
CLASS_WEIGHTS = np.array([0.51167192, 10.731 / 10])
