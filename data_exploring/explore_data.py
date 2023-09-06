import pandas as pd
import cv2
import albumentations as A
from collections import Counter

img = cv2.imread('../data/train/000001.jpg')
print(img.shape)
img = cv2.resize(img, (224, 224))
print(img.shape)
