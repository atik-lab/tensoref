import tensorflow as tf


class LearningRNN:
    """
    WARNING: THIS IS NOT WORKING PROPERLY
    LSTM is a recurrent neural network with long short term memory.
    http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
    https://gist.github.com/da-steve101/31693ebfa1b451562810d8644b788900
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    def __init__(self, data, target, n_hidden=200):
        self.data = data
        self.target = target
        self.n_hidden = n_hidden  # too high a value may lead to overfitting or a too low value may yield poor results
        self._prediction = None
        self._optimize = None
        self._error = None

    def prediction(self):
        if self._prediction is None:
            # create the RNN cell
            cell = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)
            # get the state at the end of the dynamic run as a return value but we discard it because every time we
            # look at a new sequence, the state becomes irrelevant for us
            val, state = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)
            # transpose the output to switch batch size with sequence size
            val = tf.transpose(val, [1, 0, 2])
            # take the values of outputs only at sequence’s last input
            last = tf.gather(val, int(val.get_shape()[0]) - 1)
            # apply the final transformation to the outputs of the LSTM and map it to the 21 output classes
            weight = tf.Variable(tf.truncated_normal([self.n_hidden, int(self.target.get_shape()[1])]))
            bias = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))
            # after multiplying the output with the weights and adding the bias, we will have a matrix with a
            # variety of different values for each class. What we are interested in is the probability score
            # for each class i.e the chance that the sequence belongs to a particular class. Calculate the softmax
            # activation to give us the probability scores.
            self._prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return self._prediction

    def optimize(self):
        if self._optimize is None:
            # calculate the cross entropy loss (more details here) and use that as our cost function.
            # this is the function that we are trying to minimize
            cross_entropy = -tf.reduce_sum(self.target * tf.log(tf.clip_by_value(self.prediction(), 1e-10, 1.0)))
            # choose AdamOptimzer and we set minimize to the function that shall minimize the cross_entropy loss
            # that we calculated previously
            optimizer = tf.train.AdamOptimizer()
            self._optimize = optimizer.minimize(cross_entropy)
        return self._optimize

    def error(self):
        if self._error is None:
            # this error is a count of how many sequences in the test dataset were classified incorrectly.
            # this gives us an idea of the correctness of the model on the test dataset
            mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction(), 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return self._error
