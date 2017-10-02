import numpy as np 
import xgboost as xgb 
import pandas as pd

#dataset path
dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_users_2.csv"
test_path = dataset_folder_path + "test_users.csv"

#column names
country_destination_name = 'country_destination'
id_name = 'id'
date_first_booking_name = "date_first_booking"
date_account_created_name = "date_account_created"
timestamp_first_active_name = "timestamp_first_active"
gender_name = "gender"
signup_method_name = 'signup_method'
signup_flow_name = 'signup_flow'
language_name = 'language'
affiliate_channel_name = 'affiliate_channel'
affiliate_provider_name = 'affiliate_provider'
first_affiliate_tracked_name = 'first_affiliate_tracked'
signup_app_name = 'signup_app'
first_device_type_name = 'first_device_type'
first_browser_name = 'first_browser'

#nominal attribute
nominal_column_list = [gender_name, signup_method_name, signup_flow_name, language_name, affiliate_channel_name, affiliate_provider_name, first_affiliate_tracked_name, signup_app_name, first_device_type_name, first_browser_name]

#date attribute to be separated
date_column = [date_account_created_name, timestamp_first_active_name]

#load dataset into dataframe
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
label_train = df_train[country_destination_name].values

# concat dataset into a dataframe
df_train = df_train.drop([country_destination_name], axis=1)
id_test = df_test[id_name].values #id for test
train_size = df_train.shape[0] #for separeting train and test
df_complete = pd.concat((df_train, df_test), axis=0)

# drop unnecessary column
df_complete = df_complete.drop([id_name, date_first_booking_name], axis=1)

# fill NA with -1
df_complete = df_complete.fillna(-1)

print(df_complete.info())

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


