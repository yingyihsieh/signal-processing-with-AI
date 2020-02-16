import numpy as np
import tensorflow as tf
from dl_data import data_processing,data_loading
from model_dl import CNN_model,lstm_rnn
from dl_visu import acc_line,loss_line
from sklearn.metrics import confusion_matrix

# np.random.seed(1)
# tf.set_random_seed(42)
file = './HARDataset'
feature_train, label_train, feature_test, label_test, label_test_oneh= data_loading(file)
x_traindata, x_testdata = data_processing(feature_train, feature_test, label_train)

batch_size=64
batch=len(label_train) // batch_size
epochs=21
lr = 0.0008
n_step=x_traindata.shape[1]
n_input=x_traindata.shape[2]
n_class=label_train.shape[1]
# after 4level pool for ksize:2,strider:2
flat_shape=8*144
# for lstm_rnn hidden
hidden_uni=128

def variable(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        std=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', std)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)
# placehoder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,n_step,n_input])
    y = tf.placeholder(tf.float32,[None,n_class])
with tf.name_scope('drop'):
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer'):
# w&b
    with tf.name_scope('weight'):
        weight = {
            'cnn': tf.Variable(tf.random_normal([flat_shape, n_class])),
            'in': tf.Variable(tf.random_normal([n_input, hidden_uni])),
            'out': tf.Variable(tf.random_normal([hidden_uni, n_class]))
            }
        variable(weight['cnn'])
        variable(weight['in'])
        variable(weight['out'])
    with tf.name_scope('biase'):
        biase = {
            'cnn': tf.Variable(tf.constant(0.01, shape=[n_class])),
            'in': tf.Variable(tf.constant(0.1, shape=[hidden_uni,])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_class, ]))
        }
        variable(biase['cnn'])
        variable(biase['in'])
        variable(biase['out'])
# chose one model to predict
with tf.name_scope('weight_plus_biase'):
    # pred = lstm_rnn(x, weight, biase)
    pred = CNN_model(x, weight, biase,keep_prob)

with tf.name_scope('loss'):
    # cost function:softmax_cross_entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar('loss', cost)

with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_predict'):
    # return index when predict label same with y_label
        correct_pre = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pre, dtype=tf.float32))
        tf.summary.scalar('accuracy', accuracy)
with tf.name_scope('predict_result'):
    # return index of predict label
    pred_result = tf.argmax(pred, 1)

# merge all points
merge = tf.summary.merge_all()

# initial
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# set load path
# train_writer = tf.summary.FileWriter('logs/train',sess.graph)
# test_writer = tf.summary.FileWriter('logs/test',sess.graph)

# save every accuracy in one epoch
train_acc=[]
# save train avg accuracy per epoch
epo_acc=[]
# save test avg accuracy per epoch
test_epo_acc=[]
# save train every loss in one epoch
train_loss=[]
# save train avg loss per epoch
epo_loss=[]
# save test avg loss per epoch
test_loss=[]

# start train and predict
for e in range(epochs):
    for i in range(batch):
        # send a batch to train
        step = i * batch_size
        batch_x = x_traindata[step:step + batch_size, :, :]
        batch_y = label_train[step:step + batch_size, :]
        loss, _, acc= sess.run([cost, train, accuracy], feed_dict={x: batch_x,
                                                                   y: batch_y,
                                                                   keep_prob: 0.7})
        train_acc.append(acc)
        train_loss.append(loss)
    # print('epoch:',e,"Train acc: {:.6f}".format(acc))
    #train_writer.add_summary(sum, e)

    # calculate accuracy per epoch
    epo_acc.append(np.mean(train_acc[e*batch:(e+1)*batch]))
    epo_loss.append(np.mean(train_loss[e*batch:(e+1)*batch]))
    #test_epo_acc.append(np.mean(epo_acc2))
    test_acc, te_loss, pred_res = sess.run([accuracy, cost,pred_result], feed_dict={x: x_testdata,
                                                                                 y: label_test_oneh,
                                                                                 keep_prob: 1.0})

    test_epo_acc.append(test_acc)
    test_loss.append(te_loss)

    # # index in pre_array(index of one hot code label)-->ture value label
    # # ex: label:1 ↔ one hot:100000 ↔ index:0
    pre_array=np.array(pred_res)
    pre_array+=1
    cf_matrix=confusion_matrix(label_test,pre_array)
    print('epoch:',e)
    print('confuse matrix:\n',cf_matrix)


# accuracy visual
acc_line(epochs,epo_acc,test_epo_acc)
# loss visual
loss_line(epochs,epo_loss,test_loss)

