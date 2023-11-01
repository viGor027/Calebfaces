import pandas as pd
import cv2
from collections import Counter
from src.constants import CSV_PATH
import os


df_train = pd.read_csv(os.path.join(CSV_PATH, 'list_attr_celeba.csv'))
print(df_train.dtypes)

print(Counter(df_train['Bald']))
