import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class fcnet():
    def __init__(self, sesh, layer_sizes, activation = tf.nn.relu):
        self.sesh = sesh
        self.activation = activation
        self.layer_sizes = layer_sizes
        self.num_layers = len(self.layer_sizes)

        self.input = tf.placeholder(tf.float32, shape=[None, layer_sizes[0]])
        self.labels = tf.placeholder(tf.float32, shape=[None, layer_sizes[-1]])
        self.output = self.build()

        self.optimiser = self.initialise_optimiser()
        self.accuracy = self.initialise_accuracy()

        self.sesh.run(tf.global_variables_initializer())

    def initialise_optimiser(self):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.output))
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    def initialise_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.output,1), tf.argmax(self.labels,1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

    def build(self):
        x = self.input
        for i in range(self.num_layers - 2):
            x = self.fc_layer(x, self.layer_sizes[i], self.layer_sizes[i+1], activation = self.activation)

        return self.fc_layer(x, self.layer_sizes[-2], self.layer_sizes[-1], tf.identity)
    
    def fc_layer(self, x, in_size, out_size, activation):
        W = tf.Variable(tf.random_normal([in_size,out_size], stddev = 0.1))
        b = tf.Variable(tf.constant(.05, shape=[out_size]))
        y = tf.matmul(x, W) + b
        return activation(y)

    def fit(self, x, y):
        return self.sesh.run([self.optimiser], feed_dict = {self.input: x, self.labels: y})

    def get_accuracy(self, x, y):
        return self.sesh.run([self.accuracy], feed_dict = {self.input: x, self.labels: y})[0]


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# tf.summary.scalar("loss", cross_entropy)
# tf.summary.scalar("train set accuracy", accuracy)
# merged = tf.summary.merge_all()
# logs_path = 'tf_logs'

with tf.Session() as sess:
    net = fcnet(sess, [784, 500, 10])
    # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(5000):
        batch = mnist.train.next_batch(100)
        net.fit(batch[0], batch[1])
        # _, summary = sess.run([train_step, merged], feed_dict={x: batch[0], y_: batch[1]})
        if not i % 100:
            # summarys_writer.add_summary(summary, i)
            print('batch accuracy: {:.2f}, validation accuracy: {:.2f}'.format(
                net.get_accuracy(batch[0], batch[1]), 
                net.get_accuracy(mnist.validation.images, mnist.validation.labels))
            )

    print('test accuracy: {:.2f}'.format(net.get_accuracy(mnist.test.images, mnist.test.labels)))