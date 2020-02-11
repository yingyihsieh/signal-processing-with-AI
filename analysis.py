import pandas as pd
import numpy as np
import tensorflow as tf
from dl_data import data_processing,data_loading
from model_dl import CNN_model,lstm_rnn

np.random.seed(42)
# tf.set_random_seed(42)
file = './HARDataset'
feature_train, label_train, feature_test, label_test, label_test_oneh= data_loading(file)
x_traindata, x_testdata = data_processing(feature_train, feature_test, label_train)

batch_size=64
epochs=1
lr = 0.001
n_step=x_traindata.shape[1]
n_input=x_traindata.shape[2]
n_class=y_traindata.shape[1]
# after 4level pool for ksize:2,strider:2
flat_shape=8*144
# for lstm_rnn hidden
hidden_uni=128

# placehoder
x = tf.placeholder(tf.float32,[None,n_step,n_input])
y = tf.placeholder(tf.float32,[None,n_class])
keep_prob = tf.placeholder(tf.float32)

# w&b
weight = {
    'cnn': tf.Variable(tf.random_normal([flat_shape, n_class])),
    'in': tf.Variable(tf.random_normal([n_input, hidden_uni])),
    'out': tf.Variable(tf.random_normal([hidden_uni, n_class]))
}
biase = {
    'cnn': tf.Variable(tf.constant(0.01, shape=[n_class])),
    'in': tf.Variable(tf.constant(0.1, shape=[hidden_uni,])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_class, ]))
}

# chose one model to predict
pred = lstm_rnn(x, weight, biase)
pred = CNN_model(x, weight, biase)
# cost function:softmax_cross_entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

train = tf.train.AdamOptimizer(lr).minimize(cost)

# return index when predict label same with y_label
correct_pre = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, dtype=tf.float32))
# return index of predict label
pred_result = tf.argmax(pred, 1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for i in range(0, len(y_traindata) // batch_size):
        # send a batch to train
        step = i * batch_size
        batch_x = x_traindata[step:step + batch_size, :, :]
        batch_y = y_traindata[step:step + batch_size, :]
        loss, _, acc = sess.run([cost, train, accuracy], feed_dict={x: batch_x,
                                                                    y: batch_y,
                                                                    keep_prob:0.7})
        print("Train acc: {:.6f}".format(acc))

pre_list=[]
for l in range(0, len(label_test) // batch_size):
    # send a batch of test data
    step = l * batch_size
    testbatch_x = x_testdata[step:step + batch_size, :, :]
    testbatch_y = label_test_oneh[step:step + batch_size, :]
    # get predict accuracy
    batch_acc,pred_res= sess.run([accuracy,pred_result], feed_dict={x: testbatch_x,
                                                                    y: testbatch_y,
                                                                    keep_prob: 1.0})
    print("Test acc: {:.6f}".format(batch_acc))
    # add predict result to pre_list
    for k in pred_res:
        pre_list.append(k)

# print(pre_list)
# #
# pre_array = np.array(pre_list).reshape((-1,1))
# # index in pre_array(index of one hot code label)-->ture value label
# # ex: label:1 ↔ one hot:100000 ↔ index:0
# pre_array += 1

# cf_matrix = confusion_matrix(te_label[0:pre_array.shape[0]],pre_array)
# print(cf_matrix)
