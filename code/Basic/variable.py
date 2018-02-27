import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(x))
writer.close() # close the writer when you’re done using it


a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.multiply(a, b, name='dot_product')
with tf.Session() as sess:
	print(sess.run(x))
# >> [[0 2]
#	 [4 6]]

#tf.zeros(shape, dtype=tf.float32, name=None)
#creates a tensor of shape and all elements will be zeros (when ran in session)

x = tf.zeros([2, 3], tf.int32)
y = tf.zeros_like(x, optimize=True)
print(y)
print(tf.get_default_graph().as_graph_def())
with tf.Session() as sess:
	y = sess.run(y)


with tf.Session() as sess:
	print(sess.run(tf.linspace(10.0, 13.0, 4)))
	print(sess.run(tf.range(5)))
	for i in np.arange(5):
		print(i)

samples = tf.multinomial(tf.constant([[1., 3., 1]]), 5)

with tf.Session() as sess:
	for _ in range(10):
		print(sess.run(samples))

t_0 = 19
x = tf.zeros_like(t_0) # ==> 0
y = tf.ones_like(t_0) # ==> 1

with tf.Session() as sess:
	print(sess.run([x, y]))

t_1 = ['apple', 'peach', 'banana']
x = tf.zeros_like(t_1) # ==> ['' '' '']
y = tf.ones_like(t_1) # ==> TypeError: Expected string, got 1 of type 'int' instead.

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]
x = tf.zeros_like(t_2) # ==> 2x2 tensor, all elements are False
y = tf.ones_like(t_2) # ==> 2x2 tensor, all elements are True
with tf.Session() as sess:
	print(sess.run([x, y]))

with tf.variable_scope('meh') as scope:
	a = tf.get_variable('a', [10])
	b = tf.get_variable('b', [100])

writer = tf.summary.FileWriter('test', tf.get_default_graph())


x = tf.Variable(2.0)
y = 2.0 * (x ** 3)
z = 3.0 + y ** 2
grad_z = tf.gradients(z, [x, y])
with tf.Session() as sess:
	sess.run(x.initializer)
	print(sess.run(grad_z))


# Example 1: how to run assign op
W = tf.Variable(10)
assign_op = W.assign(100)

with tf.Session() as sess:
	sess.run(W.initializer)
	print(W.eval()) # >> 10
	print(sess.run(assign_op)) # >> 100

# Example 2: tricky example
# create a variable whose original value is 2
my_var = tf.Variable(2, name="my_var")

# assign 2 * my_var to my_var and run the op my_var_times_two
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(my_var_times_two)) # >> 4
	print(sess.run(my_var_times_two)) # >> 8
	print(sess.run(my_var_times_two)) # >> 16

# Example 3: each session maintains its own copy of variables
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

# You have to initialize W at each session
sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2))) # >> 8

print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50))) # >> -42

sess1.close()
sess2.close()