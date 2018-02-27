import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#import train mnist dataset
import input_data
mnist = input_data.read_data_sets("../../data/mnist", one_hot=False)

#Parameters
learning_rate = 0.1
num_steps = 1000
batch_size =128
display_step = 100

#Network Parameters
n_hidden1 = 256
n_hidden2 = 256
num_input = 784
num_classes = 10

#Define the input function  for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True)

#Define the neural_network
def neural_net(x_dict):
    x = x_dict['images']
    layer1 = tf.layers.dense(x, n_hidden1)
    layer2 = tf.layers.dense(layer1, n_hidden2)
    out_layer = tf.layers.dense(layer2, num_classes)
    return out_layer

#define the model function
def model_fn(features, labels, mode):
    logits = neural_net(features)
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    #Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    #Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    #TF Estimators requires to return a EstimatorSpec
    estim_specs = tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op}
    )
    return estim_specs

model = tf.estimator.Estimator(model_fn)
model.train(input_fn, steps=num_steps)

#Evaluate the Model
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.test.images},
    y=mnist.test.labels,
    batch_size = batch_size,
    shuffle=False
)

model.evaluate(input_fn)

#Predict single images
n_images = 4
test_images = mnist.test.images[:n_images]
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_images}, shuffle=False
)

preds = list(model.predict(input_fn))

#Display
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction:", preds[i])
