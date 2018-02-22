import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
sess = tf.Session()
print (sess.run(a+b))
add = tf.add(a, b)
print (sess.run(add))
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
mul = tf.multiply(a, b)
print (sess.run(mul, feed_dict={a:1, b:2}))

mat1 = tf.constant([[3., 3.]])  #1x2 matrix
mat2 = tf.constant([[2.], [2.]]) #2x1 matrix
product = tf.matmul(mat1, mat2)
print (sess.run(product))

