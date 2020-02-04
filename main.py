from class_set import Save_and_load, Deal_with_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



if __name__ == '__main__':
    # set basic info of data and sql grammer by pandas to save and load
    data_name = ['body_acc_x_', 'body_acc_y_',
                 'body_acc_z_', 'body_gyro_x_',
                 'body_gyro_y_', 'body_gyro_z_',
                 'total_acc_x_', 'total_acc_y_',
                 'total_acc_z_']
    data_path = 'D:/DeepLearning/main/HARDataset/'

    sql = '''
        select * from
        '''
    data_type = 'train'

    sl = Save_and_load(data_path, data_name, data_type, sql)
    # connect to mysql
    sl.engine_create()
    # save original data
    # sl.save_to_sql()
    # load original from mysql, and set the train data and label
    sl.load_from_sql()
    train_data, train_label = sl.data, sl.label

    # load test data
    data_type = 'test'
    sl = Save_and_load(data_path, data_name, data_type, sql)
    sl.engine_create()
    # save original data
    # sl.save_to_sql()
    sl.load_from_sql()
    test_data, test_label = sl.data, sl.label

    dd = Deal_with_data(train_data)
    train = dd.data_set_array()

    dd = Deal_with_data(test_data)
    test = dd.data_set_array()

    # random forest model
    rf = RandomForestClassifier(n_estimators=180, random_state=123, min_samples_leaf=2)
    rf.fit(train, train_label)

    predict = rf.predict(test)
    predict = predict.reshape([-1, 1])
    # show accuracy
    accuracy = accuracy_score(test_label, predict)
    print(accuracy)

    # show confusion_matrix between label and predict
    c_matrix = confusion_matrix(test_label, predict)
    print(c_matrix)
