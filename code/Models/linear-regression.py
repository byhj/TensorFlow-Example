import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Parameters
rng = np.random
learn_rate = 0.01
train_epochs = 1000
display_step = 50

#Training Data
trainX = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
trainY = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
sampleNum = trainX.shape[0]

#tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

#linear model
pred = tf.add(tf.multiply(X, W), b)

#Gradient descent find the  para
cost = tf.reduce_sum(tf.pow(pred-Y,2)) / (2*sampleNum)
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

#Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

#Start training

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(train_epochs):
        for (x, y) in zip(trainX, trainY):
            sess.run(optimizer, feed_dict={X:x, Y:y})

        #display logs per epoch
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X:trainX, Y:trainY})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9}f".format(c), \
                  "W=", sess.run(W), "b=", sess.run(b))

    print ("Optimization Finished")
    trainCost = sess.run(cost, feed_dict={X:trainX, Y:trainY})
    print ("Training Cost=", trainCost, "W =", sess.run(W), "b=", sess.run(b), '\n')

    #Graphic display
    plt.plot(trainX, trainY, 'ro', label='original data')
    plt.plot(trainX, sess.run(W)*trainX + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()


