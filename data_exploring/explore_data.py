import pandas as pd
import cv2
import albumentations as A
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

df_train = pd.read_csv('../data/csv/train_cat.csv')
print(df_train.dtypes)

rus = RandomUnderSampler(random_state=0)

X_train, y_train = rus.fit_resample(df_train[['image_id']], df_train['Bald'])

print(X_train.head().tolist())

print(Counter(df_train['Bald']))
print(Counter(y_train))
