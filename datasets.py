import os
import pandas as pd
import cv2
import albumentations as A
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


TRAIN_SET_PATH = 'data/train'
VALID_SET_PATH = 'data/valid'
TEST_SET_PATH = 'data/test'

CSV_DICT = {
    TRAIN_SET_PATH: 'data/csv/train_cat.csv',
    VALID_SET_PATH: 'data/csv/valid_cat.csv',
    TEST_SET_PATH: 'data/csv/test_cat.csv'
}

transform_1 = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GridDistortion(p=0.5),
    A.HueSaturationValue(p=0.5)
])

transform_2 = A.Compose([
    A.Perspective(p=0.5),
    A.RGBShift(p=1)
])

transform_3 = A.Compose([
    A.PiecewiseAffine(p=1),
    A.CLAHE(p=0.5)
])

TRANSFORMATIONS = [transform_1, transform_2, transform_3]

y_train = pd.read_csv(CSV_DICT[TRAIN_SET_PATH])
classes = [0, 1]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train["Bald"])


def get_gen(imgs_path):
    csv = pd.read_csv(CSV_DICT[imgs_path])

    def gen():
        for img_fname in os.listdir(imgs_path):
            img = cv2.imread(imgs_path + '/' + img_fname)
            img_num = int(img_fname[:-4])
            category = csv.loc[csv['img_num'] == img_num, 'Bald'].reset_index(drop=True).iloc[0]
            if imgs_path == TRAIN_SET_PATH:
                for transformation in TRANSFORMATIONS:
                    transformed = transformation(image=img)['image']
                    yield transformed, (category, ), (class_weights[category], )
            yield img, (category,), (class_weights[category],)
    return gen


x_shape = (224, 224, 3)
y_shape = (1,)
w_shape = (1, )

x_type = tf.float32
y_type = tf.int8
w_type = tf.float32

batch_size = 128

train_ds = tf.data.Dataset.from_generator(get_gen(TRAIN_SET_PATH), output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type),
         tf.TensorSpec(shape=w_shape, dtype=w_type)))

train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(batch_size)


val_ds = tf.data.Dataset.from_generator(get_gen(VALID_SET_PATH), output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type),
         tf.TensorSpec(shape=w_shape, dtype=w_type)))

val_ds = val_ds.batch(batch_size)


test_ds = tf.data.Dataset.from_generator(get_gen(TEST_SET_PATH), output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type),
         tf.TensorSpec(shape=w_shape, dtype=w_type)))

test_ds = test_ds.batch(batch_size)
