import tensorflow as tf
from src.model.model import model
from src.data import train_ds, val_ds

tensorboard_callback = tf.keras.callbacks.TensorBoard('tensorboard/')
es = tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)

metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.FalseNegatives(name='FN'),
    tf.keras.metrics.FalsePositives(name='FP'),
    tf.keras.metrics.TrueNegatives(name='TN'),
    tf.keras.metrics.TruePositives(name='TP'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

nadam = tf.keras.optimizers.Nadam(learning_rate=(0.035 / 10))  # found learning rate divided by 10

model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=metrics, weighted_metrics=[])

model.fit(train_ds, validation_data=val_ds, callbacks=[es, tensorboard_callback], epochs=200)
model.save('model.keras')
model.save('model.h5')
