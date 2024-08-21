"""Script used for finding optimal learning rate."""

# from src.model.model import model
from src.custom_objects.lr_search_callback import LearningRateLossSave
from src.data import train_ds
from src.custom_objects.layer_scale import LayerScale
import tensorflow as tf
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

model = tf.keras.models.load_model('../../model_versions/best_model_so_far_10/model.keras',
                                   custom_objects={'LayerScale': LayerScale})

lrls = LearningRateLossSave(q=1.08)

nadam = tf.keras.optimizers.Nadam(learning_rate=(0.001 / 10))

for layer in model.layers[:-40]:
    layer.trainable = False

for layer in model.layers[-40:]:
    layer.trainable = True


model.compile(loss='binary_crossentropy', optimizer=nadam, weighted_metrics=[])

model.fit(train_ds, callbacks=[lrls], epochs=100, steps_per_epoch=10)

lrls.make_chart(date_time)
