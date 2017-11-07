import pandas as pd 
from collections import Counter
import math
from data_manager.data_generator import create_preprocessed_data

#dataset path
dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_users_2.csv"
train_session_path = dataset_folder_path + "left_join_train_session.csv"
test_path = dataset_folder_path + "test_users.csv"
test_session_path = dataset_folder_path + "left_join_test_session.csv"
session_path = dataset_folder_path + "sessions.csv"
train_output_path = dataset_folder_path + "train_session_preprocessed.csv"
test_output_path = dataset_folder_path + "test_session_preprocessed.csv"
'''
df_train = pd.read_csv(test_path)
df_session = pd.read_csv(session_path)

df_merge = pd.merge(df_train, df_session, left_on='id', right_on='user_id', how='left')
df_merge.drop("user_id", axis=1, inplace = True)

df_merge.to_csv(output_path, index=False)
'''
''' df_data = pd.read_csv(train_session_path, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
print(len(df_data.values))
 '''
create_preprocessed_data(train_session_path, test_session_path, train_output_path, test_output_path, session = True)

