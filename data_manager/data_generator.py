import pandas as pd
import numpy as np
from data_constants import *

def create_data_NDF_and_non_DBF(train_path, new_train_path):
    df_train = pd.read_csv(train_path)
    df_train[country_destination_name] = df_train[country_destination_name].apply(lambda x: "non-NDF" if x != "NDF" else x)
    df_train.to_csv(new_train_path)

def create_kaggle_submission_csv(y_pred, id_test, label_encoder, file_path, num_result_per_user = 5):
    id_out = []
    label_out = []
    for i in range(len(y_pred)):
        user_id = id_test[i]
        id_out += [user_id]*num_result_per_user
        sorted_args_pos = np.argsort(y_pred[i])[::-1]
        label_name_pred = label_encoder.inverse_transform(sorted_args_pos)
        label_out += label_name_pred[:num_result_per_user].tolist()

    submission = pd.DataFrame(np.column_stack((id_out, label_out)), columns=[id_name, country_name])
    submission.to_csv(file_path, index=False)

def create_data_exclude_NDF(train_path, new_train_path):
    df_train = pd.read_csv(train_path)
    df_train = df_train[df_train[country_destination_name] != 'NDF']
    df_train.to_csv(new_train_path)

def create_preprocessed_data(train_path, test_path, train_output_path, test_output_path, session = False):
    #load dataset into dataframe
    print("read train")
    df_train = pd.read_csv(train_path)
    print("read test")
    df_test = pd.read_csv(test_path)
    
    print("concat")
    # concat dataset into a dataframe
    train_size = df_train.shape[0] #for separeting train and test
    df_complete = pd.concat((df_train, df_test), axis=0)
    print("concat complete")

    print("drop unnecessary")
    # drop unnecessary column
    df_complete = df_complete.drop([date_first_booking_name], axis=1)
    print("drop complete")

    # fill NA with 0
    df_complete = df_complete.fillna(0)
    
    # Feature Engineering
    # Separate timestamp_first_active
    tfa = np.vstack(df_complete.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
    df_complete[timestamp_first_active_name+'_year'] = tfa[:,0]
    df_complete[timestamp_first_active_name+'_month'] = tfa[:,1]
    df_complete[timestamp_first_active_name+'_day'] = tfa[:,2]
    df_complete = df_complete.drop([timestamp_first_active_name], axis=1)

    #Separate date_account_created
    dac = np.vstack(df_complete.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
    df_complete[date_account_created_name+'_year'] = dac[:,0]
    df_complete[date_account_created_name+'_month'] = dac[:,1]
    df_complete[date_account_created_name+'_day'] = dac[:,2]
    df_complete = df_complete.drop([date_account_created_name], axis=1)

    nominal_column_list = nominal_train_column_list if not session else nominal_train_session_column_list
    #create one hot encoding
    for f in nominal_column_list:
        df_complete_dummy = pd.get_dummies(df_complete[f], prefix=f)
        df_complete = df_complete.drop([f], axis=1)
        df_complete = pd.concat((df_complete, df_complete_dummy), axis=1)
    
    df_complete[age_name] = df_complete[age_name].apply(lambda x: 0 if x < 13 or x > 120 else x)

    df_completes = np.split(df_complete, [train_size], axis=0)

    df_completes[0].to_csv(train_output_path, index = False)
    df_completes[1].to_csv(test_output_path, index = False)

