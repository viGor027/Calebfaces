from src.model.model import model
from src.custom_objects.lr_callback import LearningRateLossSave
from src.data import train_ds
import tensorflow as tf

lrls = LearningRateLossSave(q=1.03)

nadam = tf.keras.optimizers.Nadam(learning_rate=(0.001 / 10))

model.compile(loss='binary_crossentropy', optimizer=nadam, weighted_metrics=[])

model.fit(train_ds, callbacks=[lrls], epochs=100, steps_per_epoch=10)

lrls.make_chart()
