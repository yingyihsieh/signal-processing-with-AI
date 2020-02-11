import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def CNN_model(x,weight,biase):
    conv1 = tf.layers.conv1d(inputs=x, filters=18, kernel_size=2, strides=1, padding='same',
                             activation=tf.nn.leaky_relu)
    avg_pool_1 = tf.layers.average_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    conv2 = tf.layers.conv1d(inputs=avg_pool_1, filters=36, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.leaky_relu)
    avg_pool_2 = tf.layers.average_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    conv3 = tf.layers.conv1d(inputs=avg_pool_2, filters=72, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.leaky_relu)
    avg_pool_3 = tf.layers.average_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

    conv4 = tf.layers.conv1d(inputs=avg_pool_3, filters=144, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.leaky_relu)
    avg_pool_4 = tf.layers.average_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

    flat = tf.reshape(avg_pool_4, (-1, flat_shape))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob)

    # Predictions
    results_dense = tf.add(tf.matmul(flat, weight['cnn']), biase['cnn'])
    result = tf.nn.softmax(results_dense)
    return result

def lstm_rnn(x,weight,biase):
    # x_in
    x = tf.reshape(x,[-1,n_input])
    x_in = tf.matmul(x,weight['in'])+biase['in']
    x_in = tf.reshape(x_in,[-1,n_step,hidden_uni])
    # cell
    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_uni,forget_bias=1.0,state_is_tuple=True)
    _init_state = lstm.zero_state(batch_size,dtype=tf.float32)
    output, states = tf.nn.dynamic_rnn(lstm,x_in,initial_state=_init_state,time_major=False)
    # out
    result = tf.matmul(states[-1],weight['out'])+biase['out']
    return result
