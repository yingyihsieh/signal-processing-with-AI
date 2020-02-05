import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

class Save_and_load():
    def __init__(self, path, name, type, sql):
        self.path = path
        self.name = name
        self.type = type
        self.sql = sql
    def engine_create(self, user='root', password='hsieh1205', host='localhost', db='demo'):

        self.engine = create_engine(str(r"mysql+pymysql://%s:" + '%s' + "@%s/%s") %
                               (user, password, host, db), echo=True)

    def save_to_sql(self):
        for name in self.name:
            read_data = pd.read_csv(self.path + self.type + '/Inertial Signals/' + name +
                                    self.type + '.txt', header=None, delim_whitespace=True)
            read_data.to_sql(name + self.type, con=self.engine, if_exists='replace', index=False)

        read_label = pd.read_csv(self.path + self.type + '/y_' +
                                 self.type + '.txt', header=None, delim_whitespace=True)
        read_label.to_sql('y_' + self.type, con=self.engine, if_exists='replace', index=False)

    def load_from_sql(self):
        data_list = []

        for name in self.name:
            df = pd.read_sql_query(self.sql + name + self.type, self.engine)
            data_list.append(df.as_matrix())
        data_list = np.asarray(data_list)
        self.data = np.transpose(data_list, (1, 2, 0))

        df_label = pd.read_sql_query(self.sql + 'y_' + self.type, self.engine)
        self.label = np.asarray(df_label)

class Deal_with_data():
    def __init__(self, data):
        self.data = data

    def data_statistic1(self, index):
        loop_num = self.data.shape[0]

        mean_list = []
        std_list = []
        max_list = []
        min_list = []
        medi_list = []
        range_list = []

        for k in range(loop_num):
            mean_list.append(np.mean(self.data[k, :, index]))
            std_list.append(np.std(self.data[k, :, index]))
            max_list.append(np.max(self.data[k, :, index]))
            min_list.append(np.min(self.data[k, :, index]))
            medi_list.append(np.median(self.data[k, :, index]))
            range_list.append((np.max(self.data[k, :, index]) -
                                np.min(self.data[k, :, index])))

        data_mean = np.asarray(mean_list)
        data_std = np.asarray(std_list)
        data_max = np.asarray(max_list)
        data_min = np.asarray(min_list)
        data_medi = np.asarray(medi_list)
        data_range = np.asarray(range_list)

        data_array = np.hstack((data_mean, data_std, data_max,
                                    data_min, data_medi, data_range))

        return data_array

    def data_set_array(self):
        data_index = self.data.shape[-1]
        row_num = self.data.shape[0]
        set_array = np.zeros([row_num,
                                  6 * data_index])

        for i in range(data_index):
            set_array[:, 6 * i] = self.data_statistic1(i)[0:row_num]
            set_array[:, (6 * i + 1)] = self.data_statistic1(i)[row_num:2 * row_num]
            set_array[:, (6 * i + 2)] = self.data_statistic1(i)[2 * row_num:3 * row_num]
            set_array[:, (6 * i + 3)] = self.data_statistic1(i)[3 * row_num:4 * row_num]
            set_array[:, (6 * i + 4)] = self.data_statistic1(i)[4 * row_num:5 * row_num]
            set_array[:, (6 * i + 5)] = self.data_statistic1(i)[5 * row_num:6 * row_num]
        return set_array

class Visual_table():

    def balance_activities(self,label):
        act, act_counts = np.unique(label, return_counts=True)
        for i in range(len(act_counts)):
            act_counts[i] = act_counts[i] / label.shape[0] * 100

        plt.figure()
        plt.bar(act, act_counts, width=0.3, facecolor='red', label='train')
        plt.xlabel('activity')
        plt.ylabel('percent(%)')
        plt.legend()
        plt.show()

    def box_activities(self,data,label,data_name):
        df1=pd.DataFrame()
        for i in range(len(data_name)):
            df1[data_name[i]]= data[:,6*i]
        df1['label']=label
        label_map={1:'WALKING',2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',
                   4:'SITTING',5:'STANDING',6:'LAYING'}
        df1['label']=df1['label'].map(label_map)
        fig = plt.figure(figsize=(15, 12))
        fig.subplots_adjust(wspace=0.4, hspace=0.8, top=0.95)
        count = 0
        for name in data_name:
            plt.subplot(3, 3, count + 1)
            sns.boxplot(x='label', y=name, data=df1)
            plt.xticks(rotation=60)
            count += 1
        plt.show()

    def plot_confusion_matrix(self, cm, classes, normalize=False,
                              title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')