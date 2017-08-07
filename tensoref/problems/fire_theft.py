import numpy as np
import matplotlib.pyplot as plt
import xlrd
import tensorflow as tf

from tensoref.config import CONFIG
from tensoref.models.linear_regression import LinearRegression
from tensoref.models.polynomial_regression import PolynomialRegression

DATA_FILE = 'slr05.xls'


def run_linear(show_plots=True, n_epoch=100):
    """
    Download the data from:
    http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/excel/slr05.xls

    :param show_plots: show plots True or False
    :param n_epoch: how many epoch
    :return:
    """
    book = xlrd.open_workbook(CONFIG.get('general', 'data-dir') + DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = sheet.nrows - 1

    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
    model = LinearRegression(x, y)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(CONFIG.get('general', 'data-dir') + 'fire-theft-linear-graphs', sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(n_epoch):
            total_loss = 0
            for data_x, data_y in data:
                _, l, s = sess.run([model.optimizer, model.loss, model.summary], feed_dict={x: data_x, y: data_y})
                total_loss += l
                writer.add_summary(s, global_step=i)
            print('Epoch {0}: Avg {1}'.format(i, total_loss/n_samples))
        w_value, b_value = sess.run([model.w, model.b])
        writer.close()

    # show values
    print(w_value, b_value)
    if show_plots:
        data_x = data[:, 0]
        data_y = data[:, 1]
        plt.plot(data_x, data_y, '.', label='Real data')
        plt.plot(data_x, w_value*data_x + b_value, label='Prediction')
        plt.xlabel('Fires')
        plt.ylabel('Thefts')
        plt.show()


def run_polynomial(show_plots=True, n_epoch=1000):
    """
    Download the data from:
    http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/excel/slr05.xls

    :param show_plots: show plots True or False
    :param n_epoch: how many epoch
    :return:
    """
    book = xlrd.open_workbook(CONFIG.get('general', 'data-dir') + DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = sheet.nrows - 1

    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = PolynomialRegression(x, y)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(CONFIG.get('general', 'data-dir') + 'fire-theft-polynomial-graphs', sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(n_epoch):
            total_loss = 0
            for data_x, data_y in data:
                _, l = sess.run([model.optimizer, model.loss], feed_dict={x: data_x, y: data_y})
                total_loss += l
            print('Epoch {0}: Avg {1}'.format(i, total_loss/n_samples))
        w2_value, w1_value, b_value = sess.run([model.w2, model.w1, model.b])
        writer.close()

    # show values
    print(w2_value, w1_value, b_value)
    if show_plots:
        data_x = data[:, 0]
        data_y = data[:, 1]
        plt.plot(data_x, data_y, '.', label='Real data')
        plt.plot(data_x, w2_value*data_x*data_x + w1_value*data_x + b_value, '.', label='Prediction')
        plt.xlabel('Fires')
        plt.ylabel('Thefts')
        plt.show()
