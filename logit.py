import argparse
import tensorflow as tf
from functools import wraps
from tensorflow.examples.tutorials.mnist import input_data

def doublewrap(function):
    @wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class LogisticRegression():

    def __init__(self, image, label, learning_rate):
        self.image = image
        self.label = label
        self.learning_rate = learning_rate
        self.prediction
        self.optimize
        self.error

    @define_scope
    def prediction(self):
        x = self.image
        W = tf.get_variable('W', [784, 10], dtype=tf.float32, initializer=tf.random_normal_initializer)
        b = tf.get_variable('b', [1, 10], dtype=tf.float32, initializer=tf.zeros_initializer)
        y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

        return y_hat

    @define_scope
    def optimize(self):
        error = tf.losses.sigmoid_cross_entropy(self.label, self.prediction)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        return optimizer.minimize(error)

    @define_scope
    def error(self):
        mistake = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))

        return tf.reduce_mean(tf.cast(mistake, tf.float32))

def main(batch_size, n_itter, display_step, learning_rate):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    n_itter = int(mnist.train.num_examples / batch_size)

    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    model = LogisticRegression(image, label, learning_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for itter in range(n_itter):
        total_loss = 0
        for _ in range(n_itter):
            images, labels = mnist.train.next_batch(batch_size)
            sess.run(model.optimize, {image: images, label: labels})
            error = sess.run(model.error, {image: images, label: labels})
            total_loss += error
        avg_loss = total_loss / n_itter

        if itter % display_step == 0:
            print("iteration:", '%04d' % (itter), "loss=", "{:.10f}".format(avg_loss))

    images, labels = mnist.test.images, mnist.test.labels
    error = sess.run(model.error, {image: images, label: labels})
    print('test accuracy {:2.2f}%'.format(100 * (1 - error)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='logistic regression in TensorFlow')
    parser.add_argument('-b', '--batch_size', metavar='', type=int, default=1000, help='the number of observations used per batch')
    parser.add_argument('-n', '--itter', metavar='', type=int, default=100,
                help='number of iterations used to run optimization algorithm')
    parser.add_argument('-d', '--display_step', metavar='', type=int, default=10, help='frequency at which to print progress')
    parser.add_argument('-l', '--learning_rate', metavar='', type=int, default=0.01, help='stepsize used in optimization algorithm')
    args = parser.parse_args()

    main(args.batch_size, args.itter, args.display_step, args.learning_rate)
