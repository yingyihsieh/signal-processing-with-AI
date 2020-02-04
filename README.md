# signal-processing-with-AI
1. Introduction
In this project the implementation of deep learning and machine learning on signal processing from wearable is the main idea for complex activities of human beings. It includes four main parts within AI analysis process; setting up Database with MySQL, running Data pre-processing and feature engineering, model inference and Data visualization. 

2. Implementation
There are further information and operaation below,
1.MySQL Database establishment.
- save raw data to database
- load raw data from database
- by pandas and sqlalchemy
2.Data pre-proccessing
- use numpy transfer data from DataFrame to array
- basic statistic there are mean, std, medium, max, min, range 
3.Model Inference
- random forest model 
- set n_estimators=180, random_state=123, min_sample_leaf=2 
4. Data visualization
- show accuracy score
- show confuse matrix
3. Environment setting up
- Python 3.6
- Tensorflow 1.4
- Numpy
- Pandas
- sqlalchemy
- Matplotlib
- Mysql 8.0.18

4. Further work
- save clear data to Mysql database
- load clear data from Mysql database
- show activities of balance
- show three-dimensional avg acceleration for each sensor
- test overfitting 
