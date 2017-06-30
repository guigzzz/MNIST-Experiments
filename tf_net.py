import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

def fc_layer(x, in_size, out_size, activation = 'relu'):
    W = tf.Variable(tf.random_normal([in_size,out_size], stddev = 0.1))
    b = tf.Variable(tf.constant(.05, shape=[out_size]))
    y = tf.matmul(x, W) + b
    if activation == 'relu':
        return tf.nn.relu(y)
    elif activation == 'logit':
        return y
    else:
        print('unknown activation function')
        sys.exit()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

fc1 = fc_layer(x, 784, 500)
fc2 = fc_layer(fc1, 500, 500)
fc3 = fc_layer(fc2, 500, 500)
y = fc_layer(fc3, 500, 10, activation = 'logit')

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        if not i % 100:
            print(accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels}))


    print('')
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    

    

    