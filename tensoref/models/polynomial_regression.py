import tensorflow as tf


class PolynomialRegression:
    """
    Polynomial regression will find the best [w2], [w1] and [b] for the function [Y = w2*X^2 + w1*X + b] by minimizing
    the total loss with the training data using gradient descent.
    """
    def __init__(self, x, y):
        """
        Initialize all variables given placeholders x and y.

        :param x: tf.placeholder
        :param y: tf.placeholder
        """
        self.x = x
        self.y = y
        self.w2 = None
        self.w1 = None
        self.b = None
        self.prediction = self.create_prediction()
        self.loss = self.create_loss()
        self.optimizer = self.create_optimizer()

    def create_prediction(self):
        """
        Create the prediction with variables [w2], [w1] and [b] so that computes [Y = w2*X^2 + w1*X + b]

        :return: prediction
        """
        with tf.name_scope('prediction'):
            self.w2 = tf.Variable(0.0, name='weight_2')
            self.w1 = tf.Variable(0.0, name='weight_1')
            self.b = tf.Variable(0.0, name='bias')
            return tf.multiply(self.w2, tf.pow(self.x, 2)) + tf.multiply(self.w1, self.x) + self.b

    def create_loss(self):
        """
        Create an operation that calculates loss

        :return: loss
        """
        with tf.name_scope('loss'):
            return tf.square(self.prediction - self.y, name='loss')

    def create_optimizer(self):
        """
        Create an optimizer with an operation that minimizes loss

        :return: optimizer
        """
        return tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.loss)

    def create_summary(self):
        """
        Create a summary to visualize with Tensorboard

        :return:
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            return tf.summary.merge_all()
