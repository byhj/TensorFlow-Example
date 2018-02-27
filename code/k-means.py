import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
full_data_x = mnist.train.images

num_steps = 50
batch_size = 1024
k = 25
num_classes  = 10
num_features = 784 #image is 28x28

X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

(all_scores, cluster_idx, scores, cluster_centers_initialized,  init_op, train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0]
avg_distance  = tf.reduce_mean(scores)

init_vars = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_vars, feed_dict={X:full_data_x})
sess.run(init_op, feed_dict={X:full_data_x})

#Training
for i in range(1, num_steps+1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X:full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance:%f" % (i, d))
