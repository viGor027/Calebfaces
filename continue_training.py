import tensorflow as tf
from src.custom_objects.layer_scale import LayerScale
from src.custom_objects.lr_schedule import LearningRateSchedule
from src.data import train_ds, val_ds

metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.FalseNegatives(name='FN'),
    tf.keras.metrics.FalsePositives(name='FP'),
    tf.keras.metrics.TrueNegatives(name='TN'),
    tf.keras.metrics.TruePositives(name='TP'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

"""Unfreeze"""

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorboard/best_model_continued",
                                                      update_freq='batch')

nadam = tf.keras.optimizers.Nadam(learning_rate=0.002 / 100)

scheduler = LearningRateSchedule(initial_lr=0.002 / 100, target_lr=0.002, n_batches=1500)

model = tf.keras.models.load_model('model_versions/best_model_so_far_10/model.keras',
                                   custom_objects={'LayerScale': LayerScale})

for layer in model.layers[:-30]:
    layer.trainable = False

for layer in model.layers[-30:]:
    layer.trainable = True

model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=metrics, weighted_metrics=[])

model.fit(train_ds, validation_data=val_ds, callbacks=[tensorboard_callback, scheduler], epochs=1)

model.save("tensorboard/best_model_continued/model.keras")
scheduler.make_chart('20_08_24_22_10')
