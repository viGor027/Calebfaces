import os
import pandas as pd
import cv2
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from src.constants import CSV_DICT, CLASS_WEIGHTS, TRAIN_SET_PATH, VALID_SET_PATH, TEST_SET_PATH, TRANSFORMATIONS
from typing import Tuple, Callable
import numpy as np

transform = A.OneOf(TRANSFORMATIONS, p=0.5)


def get_gen(imgs_path: str, less_samples: bool) -> Callable[[], Tuple[np.ndarray, Tuple[int], Tuple[float]]]:
    """
        Generate batches of images and corresponding labels from the specified dataset.

        Args:
            imgs_path (str): Path to the directory containing the images.
            less_samples (bool): If True, reduce the number of samples in the dataset.

        Returns:
            Callable: A generator function that yields batches of images, labels, and class weights.

        """
    def gen() -> Tuple[np.ndarray, Tuple[int], Tuple[float]]:
        """
        Generator function for creating batches of images and labels.

        Yields:
            Tuple[np.ndarray, Tuple[int], Tuple[float]]: A tuple containing the image, label, and class weight.
        """
        df = pd.read_csv(CSV_DICT[imgs_path])
        class_col = df['Bald']
        img_fnames = df[['image_id']]
        if less_samples:
            img_fnames, _, class_col, _ = train_test_split(img_fnames, class_col,
                                                           train_size=batch_size*250,
                                                           random_state=0,
                                                           stratify=class_col)
        class_col = class_col.tolist()
        img_fnames = img_fnames['image_id'].tolist()
        for img_fname, category in zip(img_fnames, class_col):
            file_path = os.path.join(imgs_path, img_fname)
            if not os.path.exists(file_path):
                continue
            img = cv2.imread(file_path)
            img = cv2.resize(img, x_shape[:2])
            yield img, (category,), (CLASS_WEIGHTS[category],)

    return gen


def get_gen_train(less_samples: bool, oversample: bool) -> Callable[[], Tuple[np.ndarray, Tuple[int], Tuple[float]]]:
    """
    Generate batches of training images and corresponding labels from the specified dataset.

    Args:
        less_samples (bool): If True, reduce the number of samples in the dataset.
        oversample (bool): If True, oversample the dataset to balance class distribution.

    Returns:
        Callable: A generator function that yields batches of transformed images, labels, and class weights.

    """
    def gen_train() -> Tuple[np.ndarray, Tuple[int], Tuple[float]]:
        """
        Generator function for creating batches of transformed training images and labels.

        Yields:
            Tuple[np.ndarray,
                  Tuple[int],
                  Tuple[float]
                 ]: A tuple containing the transformed image, label, and class weight.

        """
        df = pd.read_csv(CSV_DICT[TRAIN_SET_PATH])
        class_col = df['Bald']
        img_fnames = df[['image_id']]
        if less_samples and not oversample:
            img_fnames, _, class_col, _ = train_test_split(img_fnames, class_col,
                                                           train_size=batch_size * 1500,
                                                           random_state=42,
                                                           stratify=class_col)
        if oversample:
            over = RandomOverSampler(random_state=0)
            img_fnames, class_col = over.fit_resample(img_fnames, class_col)

        class_col = class_col.tolist()
        img_fnames = img_fnames['image_id'].tolist()
        for img_fname, category in zip(img_fnames, class_col):
            file_path = os.path.join(TRAIN_SET_PATH, img_fname)
            if not os.path.exists(file_path):
                continue
            img = cv2.imread(file_path)
            img = cv2.resize(img, x_shape[:2])
            transformed = transform(image=img)['image']
            yield img, (category,), (CLASS_WEIGHTS[category],)

    return gen_train


x_shape = (224, 224, 3)
y_shape = (1,)
w_shape = (1,)

x_type = tf.float32
y_type = tf.float32
w_type = tf.float32

batch_size = 64

train_ds = tf.data.Dataset.from_generator(get_gen_train(True, False), output_signature=(
    tf.TensorSpec(shape=x_shape, dtype=x_type),
    tf.TensorSpec(shape=y_shape, dtype=y_type),
    tf.TensorSpec(shape=w_shape, dtype=w_type)))

train_ds = train_ds.shuffle(1_000)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(get_gen(VALID_SET_PATH, False), output_signature=(
    tf.TensorSpec(shape=x_shape, dtype=x_type),
    tf.TensorSpec(shape=y_shape, dtype=y_type),
    tf.TensorSpec(shape=w_shape, dtype=w_type)))

val_ds = val_ds.batch(batch_size)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_generator(get_gen(TEST_SET_PATH, False), output_signature=(
    tf.TensorSpec(shape=x_shape, dtype=x_type),
    tf.TensorSpec(shape=y_shape, dtype=y_type),
    tf.TensorSpec(shape=w_shape, dtype=w_type)))

test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
