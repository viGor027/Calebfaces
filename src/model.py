import tensorflow as tf
from data import x_shape, train_ds, val_ds


base_model = tf.keras.applications.ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=x_shape)

for layer in base_model.layers:
    layer.trainable = False

new_output = base_model.output
new_output = tf.keras.layers.Flatten()(new_output)
new_output = tf.keras.layers.Dense(64, activation='relu')(new_output)
new_output = tf.keras.layers.Dropout(0.3)(new_output)
# new_output = tf.keras.layers.Dense(150, activation='relu')(new_output)
# new_output = tf.keras.layers.Dropout(0.3)(new_output)
new_output = tf.keras.layers.Dense(1, activation='sigmoid')(new_output)

model = tf.keras.models.Model(inputs=base_model.input, outputs=new_output)

tensorboard_callback = tf.keras.callbacks.TensorBoard('tensorboard/')
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

model.fit(train_ds, validation_data=val_ds, callbacks=[tensorboard_callback, es], epochs=200)
model.save('model.keras')
model.save('model.h5')
