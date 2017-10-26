import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import matplotlib as plt

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Initialise the marker types and colors
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    color_Map = ListedColormap(colors[:len(np.unique(y))])  # we take the color mapping correspoding to the
    # amount of classes in the target data

    # Parameters for the graph and decision surface
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=color_Map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot samples
    X_test, Y_test = X[test_idx, :], y[test_idx]

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=color_Map(idx),
                    marker=markers[idx], label=cl
                    )

def train_validate_split(df, train_percent=.6, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:]]
    return train, validate

C_param_range = [0.001,0.01,0.1,1,10,100]

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


## Scale
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)


best_c = -1
best_score = -1


for i in C_param_range:
    # Apply logistic regression model to training data
    clf = LogisticRegression(penalty='l2', C=i, random_state=0)

    # Saving accuracy score in table
    score = cross_val_score(clf, x_train, y_train, cv=5)
    print("For C = %.3f , the accuracy is : %0.2f (+/- %0.2f)" % (i, score.mean(), score.std() * 2))

    cur_score = score.mean() - score.std() * 2
    if cur_score > best_score:
        best_score = cur_score
        best_c = i


# Train logistic regression model with best C
clf_final = LogisticRegression(penalty='l2', C=best_c, random_state=0)
clf_final.fit(x_train_std, y_train)

# Predict using model
y_pred = clf_final.predict_proba(x_test_std)

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
