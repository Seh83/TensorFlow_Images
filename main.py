import tensorflow as tf
import matplotlib.image as mp_image
import matplotlib.pyplot as plot
import os

"""TRANSPOSE IMAGE"""

filename = "dand.JPG"

image = mp_image.imread(filename)

print("Image Shape:", image.shape)
print("Image:", image)

plot.imshow(image)
plot.show()

"""Tensorflow Operations On Image"""

x = tf.Variable(image, name="x")

init = tf.global_variables_initializer()

"""FLIP IMAGE HEIGHT AND WIDTHS"""
with tf.Session() as sess:
    sess.run(init)

    # transpose = tf.transpose(x, perm=[1, 0, 2])
    transpose = tf.image.transpose_image(x)

    results = sess.run(transpose)

    print("Results.Shape:", results.shape)
    plot.imshow(results)
    plot.show()
