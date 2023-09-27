from data import test_ds
import tensorflow as tf
from layer_scale import LayerScale


model = tf.keras.models.load_model('model.h5', custom_objects={'LayerScale': LayerScale})

model.evaluate(test_ds)
