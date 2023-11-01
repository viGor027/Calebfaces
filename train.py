import tensorflow as tf
from src.model import model
from src.data import train_ds, val_ds

# tensorboard_callback = tf.keras.callbacks.TensorBoard('tensorboard/')
es = tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)

metrics = [
    tf.keras.metrics.Accuracy(name='accuracy'),
    tf.keras.metrics.FalseNegatives(name='FN'),
    tf.keras.metrics.FalsePositives(name='FP'),
    tf.keras.metrics.TrueNegatives(name='TN'),
    tf.keras.metrics.TruePositives(name='TP')
]

adam = tf.keras.optimizers.Adam(learning_rate=(0.001 / 12))  # default learning rate divided by 12

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=metrics, weighted_metrics=[])

model.fit(train_ds, validation_data=val_ds, callbacks=[es], epochs=200)
model.save('model.keras')
model.save('model.h5')
