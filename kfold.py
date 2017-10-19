import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


folds = 5


def decide_model(scores_list):
    best_model = -1
    best_mean = 0.
    for i in range(len(scores_list)):
        cur_mean = scores_list[i].mean()
        if cur_mean > best_mean:
            best_model = i + 1
            best_mean = cur_mean
    print("The best model is model number %d" % best_model)
    return i


#dataset path
dataset_folder_path = "dataset/"
train_path = dataset_folder_path + "train_users_2.csv"
test_path = dataset_folder_path + "test_users.csv"

#column names
country_destination_name = 'country_destination'
country_name = 'country'
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
age_name = 'age'

nominal_column_list = [gender_name, signup_method_name, signup_flow_name, language_name, affiliate_channel_name, affiliate_provider_name, first_affiliate_tracked_name, signup_app_name, first_device_type_name, first_browser_name]
#nominal attribute

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

print(df_complete[age_name].values)

#separate data
label_encoder = LabelEncoder()
values = df_complete.values
x_train = values[:train_size]
y_train = label_encoder.fit_transform(label_train)
x_test = values[train_size:]

# Declare classifiers
clf_xgb = xgb.XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=100, objective="multi:softprob", subsample=0.5, colsample_bytree=0.5, seed=0)
clf_logreg = LogisticRegression()

# K-fold cross validation
scores_xgb = cross_val_score(clf_xgb, x_train, y_train, cv=folds)
print("XGBoost accuracy: %0.2f (+/- %0.2f)" % (scores_xgb.mean(), scores_xgb.std() * 2))
scores_logreg = cross_val_score(clf_logreg, x_train, y_train, cv=folds)
print("Logistic Regression accuracy: %0.2f (+/- %0.2f)" % (scores_logreg.mean(), scores_logreg.std() * 2))

# Decide best model
clf_list = [clf_xgb, clf_logreg]
clf_final = clf_list[decide_model([scores_xgb, scores_logreg])]


# Train best model and do prediction
clf_final.fit(x_train, y_train)
y_pred = clf_final.predict_proba(x_test)

id_out = []
label_out = []
for i in range(len(x_test)):
    user_id = id_test[i]
    id_out += [user_id]*5
    sorted_args_pos = np.argsort(y_pred[i])[::-1]
    label_name_pred = label_encoder.inverse_transform(sorted_args_pos)
    label_out += label_name_pred[:5].tolist()

submission = pd.DataFrame(np.column_stack((id_out, label_out)), columns=[id_name, country_name])
submission.to_csv("sub.csv", index=False)
