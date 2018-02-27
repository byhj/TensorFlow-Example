import os
import tensorflow as tf

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        print(sess.run(z))


#lazy loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        print(sess.run(tf.add(x, y)))