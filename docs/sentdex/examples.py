import tensorflow as tf
import numpy
import random
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

from tensoref.config import CONFIG

'''
Deep learning
Learn here: https://www.youtube.com/watch?v=oYbVFhK_olY&index=43&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
'''


def run():
    sentiment_neural_network()


# video 8
def howto_use_network():
    lemmatizer = WordNetLemmatizer()

    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_classes = 2
    hm_data = 2000000
    batch_size = 32
    hm_epochs = 10

    x = tf.placeholder('float')
    y = tf.placeholder('float')

    current_epoch = tf.Variable(1)

    hidden_1_layer = {'f_fum': n_nodes_hl1,
                      'weight': tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'f_fum': n_nodes_hl2,
                      'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes]))}

    def neural_network_model(data):
        l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)

        output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

        return output

    saver = tf.train.Saver()

    def use_neural_network(input_data):
        prediction = neural_network_model(x)
        with open('lexicon.pickle', 'rb') as f:
            lexicon = pickle.load(f)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, "model.ckpt")
            current_words = word_tokenize(input_data.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = numpy.np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    # OR DO +=1, test both
                    features[index_value] += 1

            features = numpy.np.array(list(features))
            # pos: [1,0] , argmax: 0
            # neg: [0,1] , argmax: 1
            result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
            if result[0] == 0:
                print('Positive:', input_data)
            elif result[0] == 1:
                print('Negative:', input_data)

    use_neural_network("He's an idiot and a jerk.")
    use_neural_network("This was the best store i've ever seen.")


# video 7
# https://www.youtube.com/watch?v=6rDWwL6irG0
def sentiment_neural_network():
    with open(CONFIG.get('learning', 'data_dir') + '/sentiment_set.pickle', 'rb') as f:
        train_x, train_y, test_x, test_y = pickle.load(f)  # load from the pickle

    '''
    10 classes, 0-9 (10 digits)
    0 = [1,0,0,0,0,0,0,0,0]
    1 = [0,1,0,0,0,0,0,0,0]
    ...
    9 = [0,0,0,0,0,0,0,0,1]
    '''
    n_classes = 2
    # deep learning with 3 hidden layers of 500 nodes each
    n_nodes_hl1 = 200
    n_nodes_hl2 = 200
    n_nodes_hl3 = 200
    batch_size = 100

    x = tf.placeholder('float', [None, len(train_x[0])])
    y = tf.placeholder('float')

    def neural_network_model(data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes]))}

        # input_data*weights + biases
        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
        return output

    def train_neural_network(x):
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)  # learning rate default = 0.001

        hm_epochs = 10  # cycles of feedforward + backpropagation

        with tf.Session() as sess:
            # train the network
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

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            # check our accuracy
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

    train_neural_network(x)  # got 96,6 accuracy


# video 5 and 6
# https://www.youtube.com/watch?v=7fcWfUavO7E
# MemoryError -> You ran out of RAM!
def create_sentiment_featuresets():
    # this tutorial is awesome to apply in our project, because we could somehow know if a reddit post can
    # affect positively or negatively on trading
    # this uses the data from https://pythonprogramming.net/static/downloads/machine-learning-data/pos.txt
    # and https://pythonprogramming.net/static/downloads/machine-learning-data/neg.txt
    lemmatizer = WordNetLemmatizer()
    hm_lines = 10000000

    def create_lexicon(pos, neg):
        lexicon = []
        for fi in [pos, neg]:
            with open(fi, 'r') as f:
                contents = f.readlines()
                for l in contents[:hm_lines]:  # read lines one by one
                    all_words = word_tokenize(l.lower())  # chops the phrase into words
                    lexicon += list(all_words)  # add them in the list

        lexicon = [lemmatizer.lemmatize(i) for i in lexicon]  # stem into legitimate words (words that matter)
        w_counts = Counter(lexicon)  # dictionary for every element how many times exists: {'the':53423,'and':12353}

        # IDEA: check word before and after of each word to somehow use it and make it have meaning, that would be
        # to choose to chop in two words like: the cat is hungry: ['the cat','cat is','is hungry'] that could work
        # really well, but then might be hard to remove common words like I'm going to do next:
        l2 = []
        for w in w_counts:
            if 1000 > w_counts[w] > 50:  # keep only words that appear between 50 and 1000 times
                l2.append(w)
        print('Number of words that we keep (between 1000 and 50 occurrences):', len(l2))
        return l2

    def sample_handling(sample, lexicon, classification):
        # this will create a featureset that looks like:
        # [
        # [[0 1 0 1 1 0] [1 0]] <- a positive sentence and it's word occurrences
        # [[0 0 1 0 0 1] [0 1]] <- a negative sentence and it's word occurrences
        # ]
        featureset = []
        with open(sample, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                current_words = word_tokenize(l.lower())
                current_words = [lemmatizer.lemmatize(i) for i in current_words]
                features = numpy.zeros(len(lexicon))
                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1  # perhaps more than one occurrence of a word in a phrase
                features = list(features)
                featureset.append([features, classification])

        return featureset

    def create_feature_sets_and_labels(pos, neg, test_size=0.1):
        lexicon = create_lexicon(pos, neg)
        features = []
        features += sample_handling(pos, lexicon, [1, 0])
        features += sample_handling(neg, lexicon, [0, 1])
        random.shuffle(features)  # necessary for the neuronal network

        features = numpy.array(features)
        testing_size = int(test_size*len(features))

        train_x = list(features[:, 0][:-testing_size])  # from [[5,8],[7,9]] get [5,7], so we get the features
        train_y = list(features[:, 1][:-testing_size])  # from [[5,8],[7,9]] get [8,9], so we get the labels

        test_x = list(features[:, 0][-testing_size:])
        test_y = list(features[:, 1][-testing_size:])

        return train_x, train_y, test_x, test_y

    pos_txt = CONFIG.get('learning', 'data_dir') + '/pos.txt'
    neg_txt = CONFIG.get('learning', 'data_dir') + '/neg.txt'
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(pos_txt, neg_txt)
    with open(CONFIG.get('learning', 'data_dir') + '/sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)  # serializes data and dumps it in a file, not readable


# videos 3 and 4
# https://www.youtube.com/watch?v=BhpvH5DuVu8
def deep_net():
    # gathering data and getting it in the right format is a tedious process
    # feed forward + backpropagation = epoch
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
    '''
    10 classes, 0-9 (10 digits)
    0 = [1,0,0,0,0,0,0,0,0]
    1 = [0,1,0,0,0,0,0,0,0]
    ...
    9 = [0,0,0,0,0,0,0,0,1]
    '''
    n_classes = 10
    # deep learning with 3 hidden layers of 500 nodes each
    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_nodes_hl3 = 500
    batch_size = 100

    x = tf.placeholder('float', [None, 28*28])  # flat out 28 x 28 pixels
    y = tf.placeholder('float')

    def neural_network_model(data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([28*28, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes]))}

        # input_data*weights + biases
        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
        return output

    def train_neural_network(x):
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)  # learning rate default = 0.001

        hm_epochs = 15  # cycles of feedforward + backpropagation

        with tf.Session() as sess:
            # train the network
            sess.run(tf.global_variables_initializer())
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(int(mnist.train.num_examples/batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            # check our accuracy
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    train_neural_network(x)  # got 96,6 accuracy


# video 02
# https://www.youtube.com/watch?v=pnSBZ6TEVjY
def tf_basics():
    # build computation graph
    x1 = tf.constant(5)
    x2 = tf.constant(6)
    result = tf.multiply(x1, x2)

    print(result)

    # build the session and run
    with tf.Session() as sess:
        print(sess.run(result))
