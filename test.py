from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(layers.Activation('softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)

# pass optimizer by name: default parameters will be used
model.compile(loss='categorical_crossentropy', optimizer='adam')



# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam()

# Iterate over the batches of a dataset.
# for x, y in dataset:
#     # Open a GradientTape.
#     with tf.GradientTape() as tape:
#         # Forward pass.
#         logits = model(x)
#         # Loss value for this batch.
#         loss_value = loss_fn(y, logits)

#     # Get gradients of loss wrt the weights.
#     gradients = tape.gradient(loss_value, model.trainable_weights)

#     # Update the weights of the model.
#     optimizer.apply_gradients(zip(gradients, model.trainable_weights))
