import numpy
import logging
import tensorflow as tf

from cryptoblipblup.config import CONFIG

DEEP_NN_HIDDEN_NODES = 1024
DEEP_NN_EPOCH = 10
DEEP_NN_BATCH_SIZE = 64


def neural_network_model_2_hidden_layers(x, n_train_x, n_nodes_hl1, n_nodes_hl2, n_classes):
    """
    Prepare neural network model with 3 hidden layers.

    :param x: x placeholder
    :param n_train_x: number of elements in the train_x
    :param n_nodes_hl1: number of nodes in hidden layer 1
    :param n_nodes_hl2: number of nodes in hidden layer 2
    :param n_classes: number of classes
    :return:
    """
    logging.info('Preparing neural network model with 2 hidden layers of ' + str(DEEP_NN_HIDDEN_NODES) + ' nodes each')
    logging.info('[hm_epoch: ' + str(DEEP_NN_EPOCH) + ', batch_size: ' + str(DEEP_NN_BATCH_SIZE) + ']')

    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([n_train_x, n_nodes_hl1])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }
    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }
    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    # input_data*weights + biases
    l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    return tf.matmul(l2, output_layer['weights']) + output_layer['biases']


def train(train_x, train_y, test_x, test_y):
    """
    Train the deep neural network.

    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    """
    n_classes = 2
    n_train_x = len(train_x[0])
    n_nodes_hl1 = DEEP_NN_HIDDEN_NODES
    n_nodes_hl2 = DEEP_NN_HIDDEN_NODES
    hm_epochs = DEEP_NN_EPOCH
    batch_size = DEEP_NN_BATCH_SIZE

    x = tf.placeholder('float', [None, n_train_x])
    y = tf.placeholder('float')

    prediction = neural_network_model_2_hidden_layers(x, n_train_x, n_nodes_hl1, n_nodes_hl2, n_classes)

    logging.info('Training neural network')

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)  # learning rate default = 0.001

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = numpy.array(train_x[start:end])
                batch_y = numpy.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            saver.save(sess, CONFIG.get('general', 'data_dir') + 'model.ckpt')
            logging.info('Epoch ' + str(epoch) + ' completed out of ' + str(hm_epochs) + ' loss: ' + str(epoch_loss))

        # check our accuracy
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        logging.info('Accuracy: ' + str(accuracy.eval({x: test_x, y: test_y})))
