import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class fcnet():
    def __init__(self, layer_sizes, activation = tf.nn.relu):
        self.activation = activation
        self.layer_sizes = layer_sizes
        self.num_layers = len(self.layer_sizes)

    def build(self, x):
        for i in range(self.num_layers - 2):
            x = self.fc_layer(x, self.layer_sizes[i], self.layer_sizes[i+1], activation = self.activation)

        return self.fc_layer(x, self.layer_sizes[-2], self.layer_sizes[-1], tf.identity)
    
    def fc_layer(self, x, in_size, out_size, activation):
        W = tf.Variable(tf.random_normal([in_size,out_size], stddev = 0.1))
        b = tf.Variable(tf.constant(.05, shape=[out_size]))
        y = tf.matmul(x, W) + b
        return activation(y)


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

net = fcnet([784, 500, 10])
y = net.build(x)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("train set accuracy", accuracy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
logs_path = 'tf_logs'

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(5000):
        batch = mnist.train.next_batch(100)

        _, summary = sess.run([train_step, merged], feed_dict={x: batch[0], y_: batch[1]})
        if not i % 100:
            summary_writer.add_summary(summary, i)

    print('test accuracy: {}'.format(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})))