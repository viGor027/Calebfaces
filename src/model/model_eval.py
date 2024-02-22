"""Script used for evaluation of a model"""

from src.data import test_ds
import tensorflow as tf
from src.custom_objects.layer_scale import LayerScale


model = tf.keras.models.load_model('model.h5', custom_objects={'LayerScale': LayerScale})

model.evaluate(test_ds)
