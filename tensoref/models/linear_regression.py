import tensorflow as tf


class LinearRegression:
    """
    Linear regression will find the best [w] and [b] for the function [Y = wX + b] by minimizing the total loss
    with the training data using gradient descent.
    """
    def __init__(self, x, y):
        """
        Initialize all variables given placeholders x and y.

        :param x: tf.placeholder
        :param y: tf.placeholder
        """
        self.x = x
        self.y = y
        self.w = None
        self.b = None
        self.prediction = self.create_prediction()
        self.loss = self.create_loss()
        self.optimizer = self.create_optimizer()
        self.summary = self.create_summary()

    def create_prediction(self):
        """
        Create the prediction with variables [w] and [b] so that computes [Y = wX + b]

        :return: prediction
        """
        with tf.name_scope('prediction'):
            self.w = tf.Variable(0.0, name='weight_1')
            self.b = tf.Variable(0.0, name='bias')
            return tf.multiply(self.w, self.x) + self.b

    def create_loss(self):
        """
        Create an operation that calculates loss

        :return: loss
        """
        with tf.name_scope('loss'):
            return tf.square(self.prediction - self.y)

    def create_optimizer(self):
        """
        Create an optimizer with an operation that minimizes loss

        We could use also tf.train.GradientDescentOptimizer

        :return: optimizer
        """
        return tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def create_summary(self):
        """
        Create a summary to visualize with Tensorboard

        :return:
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            return tf.summary.merge_all()
