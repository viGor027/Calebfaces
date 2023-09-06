import tensorflow as tf
from data import x_shape


base_model = tf.keras.applications.ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=x_shape)

for layer in base_model.layers:
    layer.trainable = False

new_output = base_model.output
new_output = tf.keras.layers.Flatten()(new_output)
new_output = tf.keras.layers.Dense(200, activation='relu')(new_output)
new_output = tf.keras.layers.Dense(1, activation='sigmoid')(new_output)

model = tf.keras.models.Model(inputs=base_model.input, outputs=new_output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
