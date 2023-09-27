import pandas as pd
import cv2
from collections import Counter


df_train = pd.read_csv('../data/csv/list_attr_celeba.csv')
print(df_train.dtypes)

print(Counter(df_train['Bald']))
