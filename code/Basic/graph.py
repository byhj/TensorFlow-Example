import tensorflow as tf

#use the tensorboard to show the graph flow
a = tf.constant([2, 2], name='a')
b = tf.constant([2, 3], name='b')
x = tf.add(a, b, name="add")
y = tf.multiply(a, b, name="mul")

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    x, y = sess.run([x, y])
    print (x, y)
writer.close()

#python program.py
#tensorboard --logdir="./graph" --port 6006