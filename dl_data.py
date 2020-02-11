import numpy as np
import pandas as pd

def data_loading(file_name):
    traindata_name = ['body_acc_x_train', 'body_acc_y_train', 'body_acc_z_train',
                 'body_gyro_x_train', 'body_gyro_y_train', 'body_gyro_z_train',
                 'total_acc_x_train', 'total_acc_y_train', 'total_acc_z_train']
    input_traindata = []

    for name in traindata_name:
        read_data = pd.read_csv(file_name + '/train/Inertial Signals/' + name + '.txt', delim_whitespace=True, header=None)
        input_traindata.append(read_data.as_matrix())

    input_traindata = np.transpose(input_traindata, (1, 2, 0))

    testdata_name = ['body_acc_x_test', 'body_acc_y_test', 'body_acc_z_test',
                   'body_gyro_x_test', 'body_gyro_y_test', 'body_gyro_z_test',
                   'total_acc_x_test', 'total_acc_y_test', 'total_acc_z_test']
    input_testdata = []

    for name in testdata_name:
        read_data = pd.read_csv(file_name + '/test/Inertial Signals/' + name + '.txt', delim_whitespace=True, header=None)
        input_testdata.append(read_data.as_matrix())

    input_testdata = np.transpose(input_testdata, (1, 2, 0))

    label_traindata = pd.read_csv(file_name + '/train/y_train.txt', delim_whitespace=True, header=None)[0]
    label_traindata = pd.get_dummies(label_traindata).as_matrix()

    label_testdata = pd.read_csv(file_name + '/test/y_test.txt', delim_whitespace=True, header=None)[0]
    label_test_onehot = pd.get_dummies(label_testdata).as_matrix()
    label_test = np.asarray(label_testdata)
    return input_traindata, label_traindata, input_testdata, label_test,label_test_onehot


def data_processing(traindata, testdata, labeltrain, val_size = 0.1):
    input_train = traindata
    input_test = testdata
    label_train = labeltrain

    train_x = (input_train - np.mean(input_train, axis=0)[None, :, :]) / np.std(input_train, axis=0)[None, :, :]
    test_x = (input_test - np.mean(input_test, axis=0)[None, :, :]) / np.std(input_test, axis=0)[None, :, :]

    return train_x, test_x
