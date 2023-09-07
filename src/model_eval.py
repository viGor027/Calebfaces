from model import model
from data import test_ds, val_ds, train_ds
import tensorflow as tf

model.evaluate(test_ds)
