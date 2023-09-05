import pandas as pd
import os
import shutil

base_images_dir = 'data/img_align_celeba/'

try:
    os.mkdir('data/train')
    os.mkdir('data/valid')
    os.mkdir('data/test')
    os.mkdir('data/csv')
except FileExistsError:
    print('Directories already exist')

data = pd.read_csv('data/list_attr_celeba.csv')

category = data['Smiling']

y_train, y_valid, y_test = category.iloc[:162770], category.iloc[162770:182637], category.iloc[182637:]

for img in os.listdir('data/img_align_celeba'):
    if int(img[:-4]) < 162770:
        shutil.move(base_images_dir + img, 'data/train')
    elif 162770 <= int(img[:-4]) < 182637:
        shutil.move(base_images_dir + img, 'data/valid')
    else:
        shutil.move(base_images_dir + img, 'data/test')

y_train.to_csv('data/csv/train_cat.csv')
y_valid.to_csv('data/csv/valid_cat.csv')
y_test.to_csv('data/csv/test_cat.csv')
