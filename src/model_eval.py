from model import model
from data import test_ds
import tensorflow as tf

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
model.evaluate(test_ds)
