from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from data_constants import *

def load_preprocessed_data(train_path, test_path):
    
    #load dataset into dataframe
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    label_train = df_train[country_destination_name].values
    df_train = df_train.drop([country_destination_name], axis=1)

    # concat dataset into a dataframe
    id_test = df_test[id_name].values #id for test
    train_size = df_train.shape[0] #for separeting train and test
    df_complete = pd.concat((df_train, df_test), axis=0)

    # drop unnecessary column
    df_complete = df_complete.drop([id_name, date_first_booking_name], axis=1)

    # fill NA with -1
    df_complete = df_complete.fillna(-1)

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

    #create one hot encoding
    for f in nominal_column_list:
        df_complete_dummy = pd.get_dummies(df_complete[f], prefix=f)
        df_complete = df_complete.drop([f], axis=1)
        df_complete = pd.concat((df_complete, df_complete_dummy), axis=1)

    age_col = df_complete[age_name].values
    age_col = age_col.reshape((1,-1))
    imputer = Imputer(missing_values=-1, strategy="mean", axis=1)
    age_col = imputer.fit_transform(age_col)
    age_col = age_col.reshape((-1,))
    df_complete[age_name] = age_col

    #separate data
    label_encoder = LabelEncoder()
    values = df_complete.values
    x_train = values[:train_size]
    y_train = label_encoder.fit_transform(label_train)
    x_test = values[train_size:]

    return x_train, y_train, x_test
