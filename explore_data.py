import pandas as pd
import cv2
import albumentations as A
from collections import Counter


y_valid = pd.read_csv('data/csv/valid_cat.csv')
y_valid = y_valid[y_valid['Bald'] == 1]
print(y_valid.loc[y_valid['img_num'] == 162896, 'Bald'].reset_index(drop=True).iloc[0])
