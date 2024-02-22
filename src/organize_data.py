"""
Data Preparation Script

This script organizes the CelebA dataset into training, validation, and test sets.
It also creates corresponding CSV files containing image filenames and corresponding labels.

Directories:
- `TRAIN_SET_PATH`: Training set directory
- `VALID_SET_PATH`: Validation set directory
- `TEST_SET_PATH`: Test set directory
- `CSV_PATH`: Directory for CSV files

CSV Files:
- `list_attr_celeba.csv`: Original attribute file
- `train_cat.csv`: CSV file for training set
- `valid_cat.csv`: CSV file for validation set
- `test_cat.csv`: CSV file for test set

Usage:
1. Organizes images into training, validation, and test sets based on their index.
2. Creates CSV files for each set with image filenames and corresponding labels.

Note:
The attribute 'Bald' is used as a label, where '0' represents 'Not Bald' and '1' represents 'Bald'.
"""

import pandas as pd
import os
import shutil
from src.constants import TRAIN_SET_PATH, VALID_SET_PATH, TEST_SET_PATH, CSV_PATH, d


base_images_dir = os.path.join(d, 'data', 'img_align_celeba')

try:
    os.mkdir(TRAIN_SET_PATH)
    os.mkdir(VALID_SET_PATH)
    os.mkdir(TEST_SET_PATH)
    os.mkdir(CSV_PATH)
except FileExistsError:
    print('Directories already exist')

shutil.move(
    os.path.join(os.path.dirname(base_images_dir), 'list_attr_celeba.csv'),
    CSV_PATH)

data = pd.read_csv(os.path.join(CSV_PATH, 'list_attr_celeba.csv'))

category = data[['image_id', 'Bald']]
category['Bald'] = category['Bald'].apply(lambda x: 0 if x == -1 else 1)

df_train, df_valid, df_test = category.iloc[:162770], category.iloc[162770:182637], category.iloc[182637:]


for img in os.listdir(base_images_dir):
    if int(img[:-4]) < 162770:
        shutil.move(
            os.path.join(base_images_dir, img),
            TRAIN_SET_PATH)
    elif 162770 <= int(img[:-4]) < 182637:
        shutil.move(
            os.path.join(base_images_dir, img),
            VALID_SET_PATH)
    else:
        shutil.move(
            os.path.join(base_images_dir, img),
            TEST_SET_PATH)

df_train.to_csv(
    os.path.join(CSV_PATH, 'train_cat.csv'),
    index=False)

df_train.to_csv(
    os.path.join(CSV_PATH, 'valid_cat.csv'),
    index=False)

df_train.to_csv(
    os.path.join(CSV_PATH, 'test_cat.csv'),
    index=False)
