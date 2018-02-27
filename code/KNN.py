import numpy as np
import tensorflow as tf

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

Xtrain, Ytrain = mnist.train.next_batch(5000)
Xtest, Ytest = mnist.test.next_batch(200)

#TF Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

#use L1 Distance, top k small distance most label for pred
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
pred = tf.arg_min(distance, 0)
accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Xtest)):
        # get the nearest neighbor for most class label
        nn_index = sess.run(pred, feed_dict={xtr:Xtrain, xte:Xtest[i, :]})
        print ("Test " ,i, "Prediction:", np.argmax(Ytrain[nn_index]), "True Class:", np.argmax(Ytest[i]))
        #calc the arccuracy
        if np.argmax(Ytrain[nn_index]) == np.argmax(Ytest[i]):
            accuracy += 1./len(Xtest)
    print ("Done")
    print ("Accuray:", accuracy)