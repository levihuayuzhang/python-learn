import numpy as np
import tensorflow as tf

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure
weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

loss = tf.reduce_mean(tf.square(y - y_data))

for step in range(201):
    # forward + loss
    with tf.GradientTape() as tape:
        y = weights * x_data + biases
        loss = tf.reduce_mean(tf.square(y - y_data))

    # calculate gradients
    gradients = tape.gradient(loss, [weights, biases])

    # update parameters
    optimizer.apply_gradients(zip(gradients, [weights, biases]))

    if step % 20 == 0:
        print(step, weights.numpy(), biases.numpy(), loss.numpy())
