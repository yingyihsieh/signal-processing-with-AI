import numpy as np
import pandas as pd
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