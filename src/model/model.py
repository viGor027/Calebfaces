"""
TensorFlow Model Definition

This script defines a TensorFlow model using the ConvNeXtTiny architecture pre-trained on ImageNet.

Module Dependencies:
    - tensorflow (imported as tf)
    - src.data (imported for x_shape)

Global Constants:
    - x_shape (tuple): Shape of the input images.

Model Architecture:
    1. ConvNeXtTiny Base Model: Pre-trained on ImageNet.
    2. Freeze all layers in the base model.
    3. Flatten layer to convert feature maps to a flat vector.
    4. Dense layer with ReLU activation (8 units).
    5. Batch Normalization layer.
    6. Dense output layer with Sigmoid activation (binary classification).

Example Usage:
    - model = tf.keras.models.Model(inputs=base_model.input, outputs=new_output)
"""

import tensorflow as tf
from src.data import x_shape


base_model = tf.keras.applications.ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=x_shape)

for layer in base_model.layers:
    layer.trainable = False

new_output = base_model.output
new_output = tf.keras.layers.Flatten()(new_output)
new_output = tf.keras.layers.Dense(8, activation='relu')(new_output)
new_output = tf.keras.layers.BatchNormalization()(new_output)
new_output = tf.keras.layers.Dense(1, activation='sigmoid')(new_output)

model = tf.keras.models.Model(inputs=base_model.input, outputs=new_output)
